import inspect
import math
import os
import datetime
import time
import logging
import subprocess
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import selfies as sf
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from tqdm import tqdm

from rdkit.Chem import CanonSmiles, MolFromSmiles, SanitizeMol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from rdkit.Contrib.SA_Score import sascorer

# logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

char_list = ["H","C","N","O","F","P","S","Cl","Br","I",
"n","c","o","s",
"1","2","3","4","5","6","7","8",
"(",")","[","]",
"-","=","#","/","\\","+","@","<",">"]

char_dict = {c:i for i,c in enumerate(char_list)}

selfies_char_list = ['[=Branch2]', '[Ring1]', '[#Branch2]', '[\\NH1]', '[=O]', '[S@@+1]', '[=P@@]', '[-/Ring1]', '[=S]',
             '[=Ring1]', '[/C@@]', '[\\NH2+1]', '[\\O]', '[/Cl]', '[/C@]', '[=N+1]', '[=OH1+1]', '[/O+1]', '[#N]',
             '[=Branch1]', '[=C]', '[=N]', '[\\NH1+1]', '[P@@]', '[/S]', '[=S+1]', '[F]', '[S+1]', '[S@@]', '[=NH1+1]',
             '[/NH1-1]', '[\\S@]', '[\\N+1]', '[#N+1]', '[\\N]', '[I]', '[Branch2]', '[P@]', '[PH1]', '[CH2-1]',
             '[/C@H1]', '[Cl]', '[N+1]', '[Ring2]', '[\\O-1]', '[Br]', '[\\C@H1]', '[-/Ring2]', '[\\I]', '[=NH2+1]',
             '[C@@]', '[\\S-1]', '[\\C]', '[/N+1]', '[=PH2]', '[/S@]', '[\\S]', '[NH2+1]', '[nop]', '[NH1]', '[P]',
             '[Branch1]', '[\\Br]', '[=O+1]', '[-\\Ring1]', '[/N-1]', '[\\Cl]', '[P@@H1]', '[N]', '[=P]', '[NH1+1]',
             '[\\N-1]', '[/Br]', '[/NH1+1]', '[S]', '[N-1]', '[/NH2+1]', '[NH1-1]', '[#C]', '[C]', '[\\F]', '[/S-1]',
             '[/F]', '[/NH1]', '[=N-1]', '[NH3+1]', '[P+1]', '[=Ring2]', '[CH1-1]', '[S-1]', '[=P@]', '[/C]', '[=S@]',
             '[\\C@@H1]', '[O]', '[O-1]', '[/C@@H1]', '[#Branch1]', '[=SH1+1]', '[/O]', '[=S@@]', '[C@H1]', '[S@]',
             '[C@@H1]', '[/N]', '[C@]', '[/O-1]', '[PH1+1]', 'sos']

selfies_char_dict = dict(enumerate(selfies_char_list[:-1]))

def label2sf2smi(out_num):
    m_sf = sf.encoding_to_selfies(out_num, selfies_char_dict, enc_type='label')
    m_smi = sf.decoder(m_sf)
    m_smi = CanonSmiles(m_smi)
    return m_smi

def property_transform(y_data: np.array, mol_property: str):
    if mol_property in ["ba0", "phgdh"]:
        return np.log(-(y_data - 1))
    if mol_property == "ba1":
        return -y_data / 20.0
    if mol_property == "PlogP":
        return (y_data + 15) / 20
    if mol_property == "sas":
        return y_data / 10.0
    if mol_property == "qed":
        return y_data

def property_untransform(y_data: np.array, mol_property: str):
    if mol_property in ["ba0", "phgdh"]:
        return -np.expm1(y_data)
    if mol_property == "ba1":
        return -20.0 * y_data
    if mol_property == "PlogP":
        return 20 * y_data - 15
    if mol_property == "sas":
        return 10.0 * y_data
    if mol_property == "qed":
        return y_data
    raise ValueError(f"Molecule property `{mol_property}` is not one of ['ba0', 'ba1', 'phgdh', 'PlogP', 'sas', 'qed']")

def calculate_lnew(x_new, end_token=58):
    # Given x_new, end_idx, generate length data
    end_token_positions = np.argmax(x_new == end_token, axis=1)
    no_end_token = (end_token_positions == 0) & np.all(x_new != end_token, axis=1)
    end_token_positions[no_end_token] = x_new.shape[1]
    return end_token_positions

class MolDataset(Dataset):
    def __init__(self, Xdata, Ldata, Ydata=None, max_len=73): # 72 + start token
        self.Xdata = Xdata  # number-coded molecule
        self.Ldata = Ldata  # length of each molecule
        self.max_len = max_len
        self.Ydata = Ydata
        self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        # Add sos=108 for each sequence and this sos is not shown in char_list and char_dict as in selfies.
        mol = self.Xdata[index]
        sos = torch.tensor([108], dtype=torch.long)
        mol = torch.cat([sos, mol], dim=0).contiguous()
        l = self.Ldata[index] + 1
        L = mol.shape[0]
        if l < self.max_len:
            mask = torch.tensor([False] * (l+1) + [True] * (L-l-1))
        else:
            mask = torch.tensor([False] * l)
        y_data = {p: yd[index] for p, yd in self.Ydata.items()}
        return (mol, mask, y_data)

    def __len__(self):
        return self.len

    def sf_string(self, mol):
        mol = mol[1:].numpy()
        m_sf = sf.encoding_to_selfies(mol, selfies_char_dict, enc_type='label')
        return m_sf

    def sf2smi(self, mol):
        m_sf = self.sf_string(mol)
        m_smi = sf.decoder(m_sf)
        return m_smi

class MolTrainDataset(MolDataset):
    def __init__(self, dir, split, mol_property=[], mask_nonnegative=False, max_len=73): # 72 + start token
        # selfies encoding
        Xdata_file = f"{dir}/Xsf{split}.npy"
        self.Xdata = torch.tensor(np.load(Xdata_file), dtype=torch.long)  # number-coded molecule
        Ldata_file = f"{dir}/Lsf{split}.npy"
        self.Ldata = torch.tensor(np.load(Ldata_file), dtype=torch.long)  # length of each molecule
        self.max_len = max_len
        self.Ydata = OrderedDict()
        for p in mol_property:
            Ydata_file = f"{dir}/{p}_{split}.npy"
            y_data = torch.tensor(np.load(Ydata_file), dtype=torch.float32)
            if mask_nonnegative:
                neg_mask = y_data < 0
                self.Xdata = self.Xdata[neg_mask]                
                self.Ldata = self.Ldata[neg_mask]
                y_data = y_data[neg_mask]
            self.Ydata[p] = property_transform(y_data, p)
        self.len = self.Xdata.shape[0]

####################################################################################################################

def get_lr(it, args):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.lr * (it+1) / args.warmup_iters
    if (it >= args.warmup_iters) and (it < args.max_lr_iters):
        return args.lr
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.max_lr_iters) / (args.lr_decay_iters - args.max_lr_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return args.min_lr + coeff * (args.lr - args.min_lr)

##------------------------------------------------------------------------------------------------------------------##

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'dec_trans' in name and param.ndim >= 2:
            decay_params.append(param)
        else: # unet, mlp, {dec_trans params with ndim < 2}
            nodecay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=False) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

##--------------------------------------------------------------------------------------------------------------------##

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    model_state = model.state_dict() if model is not None else None
    return {'epoch': epoch, 'iter_num': it, 'model_state': model_state, 'optimizer_state': optim_state}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def truncate_str(s, max_width=20):
    return s[:max_width] + '...' if len(s) > (max_width + 3) else s

def log_dict_as_table(logger, data):
    # Find the longest key for padding
    key2str = OrderedDict()
    for (key, val) in data.items():
        valstr = f"{val:<8.5g}" if hasattr(val, '__float__') else val
        key2str[truncate_str(key)] = truncate_str(valstr)
    keywidth = max(map(len, key2str.keys()))
    valwidth = max(map(len, key2str.values()))
    dashes = '-' * (keywidth + valwidth + 7)

    logger.info(dashes)
    for key, val in key2str.items():
        logger.info(f"| {key}{' ' * (keywidth - len(key))} | {val}{' ' * (valwidth - len(val))} |")
    logger.info(dashes)

##--------------------------------------------------------------------------------------------------------------------##


def calc_prop(m):
    logP = MolLogP(m)
    try:
        SAS = sascorer.calculateScore(m)
    except ZeroDivisionError:
        SAS = 2.8
    QED = qed(m)
    PlogP = logP - SAS
    cycle_count = 0
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) > 6:
            PlogP -= 1
        if not (4 < len(ring) < 7):
            cycle_count += 1
    return (logP, SAS, QED, PlogP, cycle_count)

def is_valid(smi):
    m = MolFromSmiles(smi)
    if m is None or SanitizeMol(m, catchErrors=True):
        return 0
    return 1, calc_prop(m)

def smiles_to_affinity(smiles, autodock, protein_file, ligands_path='ligands_zero', outs_path='outs_zero', num_devices=torch.cuda.device_count(), ps_per_gpu=6):
    if not os.path.exists(ligands_path):
        os.mkdir(ligands_path)
    if not os.path.exists(outs_path):
        os.mkdir(outs_path)
    subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f'rm {outs_path}/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f'rm {outs_path}/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f'rm -rf {ligands_path}/*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f'rm -rf {outs_path}/*', shell=True, stderr=subprocess.DEVNULL)
    num_folders = ps_per_gpu * num_devices
    for folder in range(num_folders):
        os.mkdir(f'{ligands_path}/{folder}')
    folder = 0
    for i, hot in enumerate(tqdm(smiles, desc='preparing ligands')):
        subprocess.Popen(
            f'obabel -:"{smiles[i]}" -O {ligands_path}/{folder}/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d',
            shell=True, stderr=subprocess.DEVNULL)
        folder += 1
        if folder == num_folders:
            folder = 0
    while True:
        total = 0
        for folder in range(num_folders):
            total += len(os.listdir(f'{ligands_path}/{folder}'))
        if total == len(smiles):
            break
    time.sleep(1)
    print('running autodock..')
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -s 0 -L {ligands_path}/0/ligand0.pdbqt -N {outs_path}/ligand0', shell=True,
                       stdout=subprocess.DEVNULL)
    else:
        ps = []
        for device in range(num_devices):
            for progress in range(ps_per_gpu):
                folder = device * num_devices + progress
                # print(folder)
                ps.append(subprocess.Popen(
                    f'{autodock} -M {protein_file} -s 0 -B {ligands_path}/{folder}/ligand*.pdbqt -N {outs_path}/ -D 1',
                    shell=True, stdout=subprocess.DEVNULL))
        for p in ps:
            p.wait()
    affins = [0 for _ in range(len(smiles))]
    for file in tqdm(os.listdir(outs_path), desc='extracting binding values'):
        if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'{outs_path}/{file}').read():
            affins[int(file.split('ligand')[1].split('.')[0])] = float(
                subprocess.check_output(f"grep 'RANKING' {outs_path}/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1",
                                        shell=True).decode('utf-8').strip())
    return [min(affin, 0) for affin in affins]


def single_property_plot_ba_simple(args, epoch, y_hat, data_ba, output_dir=None):
    ba, data_ba = data_ba
    print(len(data_ba), len(y_hat))
    labels = ['data'] * len(data_ba) + ['LPT Pred'] * len(y_hat)
    logp_df = pd.DataFrame(data={'model': labels, ba: data_ba + y_hat})
    logp_df[ba] = logp_df[ba].astype(float)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4 * 1, 4.8])
    fig.suptitle(ba, fontsize=12)
    logp_plot = sns.kdeplot(data=logp_df, x=ba, hue='model', ax=ax)
    plt.savefig(os.path.join(output_dir, 'single_epoch_{:03d}.png'.format(epoch)))
    
    saved_image_path = os.path.join(output_dir, 'single_epoch_{:03d}.png'.format(epoch))
    image_pil = Image.open(saved_image_path)

    # Convert the PIL image to a numpy array
    img_data = np.array(image_pil)

    return {ba: img_data}

##--------------------------------------------------------------------------------------------------------------------##

def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id, fs_prefix='./', extra_tag=None):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(fs_prefix + 'output/' + exp_id, t)
    if extra_tag is not None:
        output_dir = os.path.join(output_dir, extra_tag)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

def setup_logger(name, output_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    ]
    
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
