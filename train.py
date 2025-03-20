#!/usr/bin/env python3

import datetime
import os
import time

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit.Chem import Draw

# Local/project imports
from lpt import LatentPromptTransformer
from args import parse_args
from utils import *

##--------------------------------------------------------------------------------------------------------------------##

def train_batch(model, data, optimizer, device, args, iter_num, epoch, logger):
    """Train model on a single batch of data."""        
    lr = get_lr(iter_num, args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x, x_mask, y = data
    x = x.to(device)
    y = {k:v.to(device) for k,v in y.items()}
    targets = x.detach().clone()
    targets[x_mask] = -1
    targets = targets[:, 1:]
    x = x[:, :-1]
    assert x.shape[-1] == targets.shape[-1]

    optimizer.zero_grad()
    # sample prior
    model.eval()
    z_samples, _ = model.infer_z(None, x, targets, y=y, step_size=args.z_step_size, debug=args.debug)
    model.train()
    # logits, loss
    z_prop = model.unet_forward(z_samples)
    _, loss = model.dec_trans(x, z_prop, targets, inference=False)
    
    mlp_error = 0
    if args.train_phase in ["finetune", "onlinelearn"]:
        for p, yval in y.items():
            y_hat = model.property_mlps[p](z_prop)
            mlp_error += mse_loss(y_hat, yval, reduction='mean')
        loss += mlp_error * args.prop_coefficient

    optimizer.zero_grad()
    loss.backward()
    if args.max_grad_norm > 0:
       torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return loss, mlp_error

def run_train(args, output_dir):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = setup_logger(__name__, output_dir)

    if args.wandb:
        import wandb
        current_date = datetime.datetime.today().strftime('%m-%d')
        with_noise = f'noise_{args.noise_factor}' if args.z_with_noise else 'without-noise'
        tag = args.extra_tag
        wandb.init(project="lpt-mol", name=f"train-{current_date}-{with_noise}-{tag}", dir=output_dir, config=args)

    dataset_kwargs = {"mol_property": args.mol_property, "mask_nonnegative": args.mask_nonnegative, "max_len": args.max_len}
    train_data = MolTrainDataset(args.data_dir, "train", **dataset_kwargs)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                             shuffle=True, drop_last=True, num_workers=0)
    
    # Only load test data if in conditional mode
    test_loader = None
    if args.train_phase == "finetune":
        test_data = MolTrainDataset(args.data_dir, "test", **dataset_kwargs)
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, num_workers=0)

    vocab_size = len(selfies_char_list)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    checkpoint_path = os.path.join(output_dir, "model")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LatentPromptTransformer(
        args, vocab_size=vocab_size,
        dec_word_dim=args.dec_word_dim,
        n_latent=args.n_latent,
        latent_dim=args.latent_dim,
        max_sequence_length=args.max_len
        )
    if args.train_from:
        logger.info('loading model from ' + args.train_from)
        state_dict = torch.load(args.train_from, map_location="cpu", weights_only=True)['model_state']
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state_dict:
                    param.copy_(state_dict[name].to(device))
                else:
                    print(f"{name} not found in model state_dict.")

    logger.info("model architecture")
    logger.info(str(model))
    optimizer = configure_optimizers(model, args.weight_decay, args.lr, (0.9, 0.95), device_type)
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    model.to(device)

    best_prior_prop_valid = 0.

    iter_num = 0
    
    # Perform initial evaluation if in conditional mode
    if args.train_phase == "finetune" and test_loader is not None:
        val_dict = eval_model(args, test_loader, model, logger, n_iter=0, output_dir=output_dir)
        if args.wandb:
            wandb.log({"val/validity": val_dict[0]})
            wandb.log({"val/uniqueness": val_dict[1]})
            wandb.log({"val/novelty": val_dict[2]})
            wandb.log({"val/loss": val_dict[3]})
            for ba, img in val_dict[4].items():    
                img = wandb.Image(img, caption="before training")
                wandb.log({f"val/{ba}": img})
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        logger.info(f'Starting epoch {epoch:d}')
        model.train()
        if args.debug and epoch > 5:
            break
        for i, data in enumerate(train_loader):
            if args.debug and i > 2:
                break
            loss, mlp_error = train_batch(model, data, optimizer, device, args, iter_num, epoch, logger)
            if i % args.print_every == 0:
                logger.info('Epoch={:4d}, Batch={:4d}/{:4d}, LR={:8.6f}, Loss={:10.4f},'.format(
                    epoch, i + 1, len(train_loader), get_lr(iter_num, args), loss))
            if args.wandb:
                wandb.log({"train/loss": loss.cpu().item()})
                wandb.log({"train/lr": get_lr(iter_num, args)})
                if args.train_phase == "finetune":
                    wandb.log({"train/reg_loss": mlp_error.cpu().item()})
                    wandb.log({"train/reconstruct_loss": (loss - mlp_error).cpu().item()})
            iter_num += 1

        epoch_train_time = time.time() - start_time
        logger.info(f'Time Elapsed: {epoch_train_time:.1f}s')

        log_dict_as_table(logger, {'Epoch': epoch, 'Mode': 'Train', 'LR': get_lr(iter_num, args), 'Epoch Train Time': epoch_train_time})
        torch.save(checkpoint_state(model, optimizer, epoch, iter_num), f"{checkpoint_path}_{epoch}.pth")

        # Run evaluation if in conditional mode
        if args.train_phase == "finetune" and test_loader is not None:
            val_dict = eval_model(args, test_loader, model, logger, epoch, output_dir=output_dir)

            if args.wandb:
                wandb.log({"val/validity": val_dict[0]})
                wandb.log({"val/uniqueness": val_dict[1]})
                wandb.log({"val/novelty": val_dict[2]})
                wandb.log({"val/loss": val_dict[3]})
                for ba, img in val_dict[4].items():    
                    img = wandb.Image(img, caption="epoch:{}".format(epoch))
                    wandb.log({f"val/{ba}": img})
            
            # Save best model
            prior_prop_valid = val_dict[0]
            if prior_prop_valid > best_prior_prop_valid:
                best_prior_prop_valid = prior_prop_valid
                torch.save(
                    checkpoint_state(model, optimizer, epoch, iter_num), 
                    os.path.join(output_dir, 'best_prior_mol') + ".pth"
                )

    if args.wandb:
        wandb.finish()


def run_online_learn(args, output_dir):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = setup_logger(__name__, output_dir)

    if args.wandb:
        import wandb
        current_date = datetime.datetime.today().strftime('%m-%d')
        with_noise = f'noise_{args.noise_factor}' if args.z_with_noise else 'without-noise'
        tag = args.extra_tag
        wandb.init(project="lpt-mol", name=f"train-{current_date}-{with_noise}-{tag}", dir=output_dir, config=args)

    dataset_kwargs = {"mol_property": args.mol_property, "mask_nonnegative": args.mask_nonnegative, "max_len": args.max_len}
    train_data = MolTrainDataset(args.data_dir, "train", **dataset_kwargs)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                             shuffle=True, drop_last=True, num_workers=0)
    test_data = MolTrainDataset(args.data_dir, "test", **dataset_kwargs)

    vocab_size = len(selfies_char_list)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    checkpoint_path = os.path.join(output_dir, "model")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LatentPromptTransformer(
        args, vocab_size=vocab_size,
        dec_word_dim=args.dec_word_dim,
        n_latent=args.n_latent,
        latent_dim=args.latent_dim,
        max_sequence_length=args.max_len
        )
    if args.train_from:
        logger.info('loading model from ' + args.train_from)
        state_dict = torch.load(args.train_from, map_location="cpu", weights_only=True)['model_state']
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state_dict:
                    param.copy_(state_dict[name].to(device))
                else:
                    print(f"{name} not found in model state_dict.")

    logger.info("model architecture")
    logger.info(str(model))
    optimizer = configure_optimizers(model, args.weight_decay, args.lr, (0.9, 0.95), device_type)
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    model.to(device)

    Xdata, Ldata, Ydata = test_data.Xdata, test_data.Ldata, test_data.Ydata
    x_old = torch.empty(0, Xdata.shape[1], dtype=torch.long)
    l_old = torch.empty(0, dtype=torch.long)  
    y_old = {p: torch.empty(0) for p in Ydata.keys()}
    latent_z_old = model.sample_p_0(Xdata[:args.batch_size // 2].to(device))

    if args.cond_substructure_smiles:
        sub_structure_sf_list = sf.split_selfies(sf.encoder(args.cond_substructure_smiles))
        swapped_dict = {v: k for k, v in selfies_char_dict.items()}
        sub_structure_emb = torch.tensor(list(map(swapped_dict.get, sub_structure_sf_list)), dtype=torch.long, device=model.device)

    iter_num = 0

    for epoch in range(args.num_epochs):
        start_time = time.time()
        logger.info(f'Starting epoch {epoch:d}')
        model.train()
        design_data = MolDataset(Xdata, Ldata, Ydata)
        design_loader = DataLoader(dataset=design_data, batch_size=args.batch_size // 2, shuffle=True, drop_last=False)
        train_data = design_data
        # only sample, don't train in the first epoch
        if epoch == 0:
            train_loader = []
        else:
            train_loader = DataLoader(dataset=design_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
        if args.debug and epoch > 5:
            break
        for i, data in enumerate(tqdm(train_loader, desc='training')):
            if args.debug and i > 2:
                break
            loss, mlp_error = train_batch(model, data, optimizer, device, args, iter_num, epoch, logger)
            if i % args.print_every == 0:
                logger.info('Epoch={:4d}, Batch={:4d}/{:4d}, LR={:8.6f}, Loss={:10.4f},'.format(
                    epoch, i + 1, len(train_loader), get_lr(iter_num, args), loss))
            if args.wandb:
                wandb.log({"train/loss": loss.cpu().item()})
                wandb.log({"train/lr": get_lr(iter_num, args)})
                if args.train_phase in ["finetune", "onlinelearn"]:
                    wandb.log({"train/reg_loss": mlp_error.cpu().item()})
                    wandb.log({"train/reconstruct_loss": (loss - mlp_error).cpu().item()})
            iter_num += 1

        epoch_train_time = time.time() - start_time
        log_dict_as_table(logger, {'Epoch': epoch, 'Mode': 'Train', 'LR': get_lr(iter_num, args), 'Epoch Train Time': epoch_train_time})
        torch.save(checkpoint_state(model, optimizer, epoch, iter_num), f"{checkpoint_path}_{epoch}.pth")

        model.eval()
        generated_samples = []
        x_new = []
        latent_z_new = torch.empty(0, *latent_z_old.shape[1:], device=device, requires_grad=False)
        for i, data in enumerate(tqdm(design_loader, desc='sampling')):
            if args.debug and i > 2:
                break
            if len(generated_samples) > 2500:
                break
            x, x_mask, y = data
            x = x.to(device)
            y = {k:v.to(device) for k,v in y.items()}
            targets = x.detach().clone()
            targets[x_mask] = -1
            targets = targets[:, 1:]
            x = x[:, :-1]
            assert x.shape[-1] == targets.shape[-1]

            x = x.shape[0]
            z_0 = latent_z_old[i * x:(i + 1) * x, :] if epoch > 0 else model.sample_p_0(x)
            # Paper Section 3.5: Computational efficiency
            z_y, _ = model.infer_z_given_y(z_0, {k:v+0.1 for k,v in y.items()}, n_iter=2, step_size=args.z_step_size)
            with torch.no_grad():
                if args.cond_substructure_smiles:
                    x = sub_structure_emb.view(1, -1).repeat(x, 1)
                x_hat = model.generate(sos_idx=108, max_new_tokens=72, x=x, z=z_y, do_sample=True, top_k=50)
                latent_z_new = torch.cat((latent_z_new, z_y), 0)
                for s in x_hat:
                    temp_sample = label2sf2smi(s[1:].cpu().numpy())
                    if temp_sample in generated_samples:
                        continue
                    generated_samples.append(temp_sample)
                    x_new.append(s.cpu().numpy())
                    if len(generated_samples) >= 2500:
                        break
        
        IDs = []
        SASs, QEDs = [], []
        filtered_samples = []
        for idx, s in enumerate(tqdm(generated_samples, desc='filtering')):
            _is_valid = is_valid(s)
            if _is_valid != 0:
                _, SAS, QED, _, cycle_count = _is_valid[1]
                # if cycle_count == 0 and (SAS < max_sa) and (QED > min_qed):
                if cycle_count == 0:
                    SASs.append(SAS)
                    QEDs.append(QED)
                    filtered_samples.append(s)
                    IDs.append(idx)
        
        num_print = 100
        x_new = np.array(x_new)[IDs]
        latent_z_new = latent_z_new[IDs, :]
        x_new = x_new[:,1:] # remove start token in data
        l_new = torch.tensor(calculate_lnew(x_new), dtype=torch.long)
        y_new = smiles_to_affinity(
            filtered_samples,
            args.autodock_executable, 
            args.protein_file, 
            ligands_path=f'ligands_zero_{args.mol_property[0]}', 
            outs_path=f'outs_zero_{args.mol_property[0]}', 
            num_devices=1, 
            ps_per_gpu=args.num_autodock_proc
        )
        y_new = -np.array(y_new)
        ind = np.argsort(y_new)
        ind = ind[-num_print:][::-1]
        kd = np.exp(-y_new[ind] / (0.00198720425864083 * 298.15)).flatten()
        mols = []
        props = []
        # print('ba     Kd     smi')
        logger.info('ba     Kd     smi')
        for i,id in enumerate(ind):
            logger.info(f"{y_new[id]:.4f}    {kd[i]:.9f}    {filtered_samples[id]}")
            mols.append(MolFromSmiles(filtered_samples[id]))
            props.append(f"{kd[i] * 10 ** 9:.5f}\n")
        fig = Draw.MolsToGridImage(mols, molsPerRow=5, legends=props)
        fig.save(f"{output_dir}/Kd_epoch{epoch}.png")
        print(x_old.shape, x_new.shape)
        Xdata = torch.cat([x_old, torch.tensor(x_new, dtype=torch.long)], dim=0)
        Ydata = {p: torch.cat([y_old[p], torch.tensor(y_new, dtype=torch.float32)]) for p in Ydata.keys()}
        Ldata = torch.cat([l_old, l_new])
        max_dataset_len = min(15000, *map(len, Ydata.values()))
        indices = torch.argsort(Ydata[[p for p in Ydata.keys() if (("ba" in p) or ("phgdh" in p))][0]], descending=True)[:max_dataset_len]
        logger.info(f'sampling {max_dataset_len:d} data')
        Xdata = Xdata[indices, :]
        Ydata = {p: yd[indices] for p, yd in Ydata.items()}
        Ldata = Ldata[indices]
        latent_z_all = torch.cat([latent_z_old, latent_z_new], dim=0)
        latent_z_old = latent_z_all[indices, :]
        x_old = Xdata
        l_old = Ldata
        y_old = Ydata

    if args.wandb:
        wandb.finish()

##--------------------------------------------------------------------------------------------------------------------##
def eval_validity(args, data_loader, model, nbatch=None, reconstruction=False, validity_only=False, output_dir=None, epoch=0):
    generated_samples = []
    properties = {p: [] for p in args.mol_property}
    model.eval()
    for data in data_loader:
        for i in range(0, len(data[0]), nbatch or len(data[0])):
            x, _, _ = data
            x = x[i:i+nbatch].cuda()
            target = x.detach().clone()
            target = target[:, 1:]
            x = x[:, :-1]
            z_0 = model.sample_p_0(x)

            # TODO:  mol_property regression using Z unet?
            if reconstruction:
                z, _ = model.infer_z(z_0, x, target, step_size=args.z_step_size)
            else:
                z = z_0
            with torch.no_grad():
                x_hat = model.generate(sos_idx=108, max_new_tokens=72, x=x.shape[0], z=z, do_sample=True, top_k=50)
                generated_samples += [label2sf2smi(s[1:].cpu().numpy()) for s in x_hat]
                z_prior = model.unet_forward(z)
                for p, p_mlp in model.property_mlps.items():
                    y_hat = p_mlp(z_prior)
                    property_pred = y_hat.detach().cpu().numpy()
                    property_pred = property_untransform(property_pred, p)
                    properties[p] += [ppv for ppv in property_pred]
    # validity
    logPs, SASs, QEDs, PlogPs = [], [], [], []
    num_valid = 0
    for s in generated_samples:
        _is_valid = is_valid(s)
        if _is_valid != 0:
            num_valid += _is_valid[0]
            # logP, SAS, QED, PlogP = _is_valid[1]
            # logPs.append(logP)
            # SASs.append(SAS)
            # QEDs.append(QED)
            # PlogPs.append(PlogP)
    validity = num_valid / len(generated_samples)

    ydata = MolTrainDataset(args.data_dir, "test", mol_property=args.mol_property, mask_nonnegative=args.mask_nonnegative, max_len=args.max_len).Ydata
    img_data = {}
    for p, pvals in ydata.items():
        data_ba = (p, property_untransform(pvals, p).numpy().tolist())
        # single_property_plot_logp(args, PlogPs, properties, data_plogp, epoch=epoch, output_dir=output_dir)
        img_data.update(single_property_plot_ba_simple(args, epoch, properties[p], data_ba, output_dir=output_dir))


    model.train()
    if validity_only:
        return validity
    # uniqueness
    unique_set = set(generated_samples)
    validity = num_valid / len(generated_samples)
    uniqueness = len(unique_set) / len(generated_samples)

    # novelty
    ZINC_file = "data/test_5.txt"
    ZINC_data = [x.strip().split()[0] for x in open(ZINC_file) if not x.startswith("#smi")]
    ZINC_set = set(ZINC_data)
    novel_list = list(unique_set - ZINC_set)
    novelty = len(novel_list) / len(generated_samples)

    return validity, uniqueness, novelty, img_data


def eval_model(args, test_loader, model, logger, n_iter=None, output_dir=None):
    total_nll_abp = 0.
    test_data_size = 0
    model.eval()
    device = next(model.parameters()).device
    for i, data in enumerate(tqdm(test_loader, leave=False)):
        if args.debug and i > 10:
            break
        x, _, y = data
        x = x.to(device)
        y = {k:v.to(device) for k,v in y.items()}
        targets = x.detach().clone()[:, 1:]
        x = x[:, :-1]

        z_samples, _ = model.infer_z(None, x, targets, step_size=args.z_step_size)
        with torch.no_grad():
            z_prop = model.unet_forward(z_samples)
            _, loss = model.dec_trans(x, z_prop, targets, inference=False) # logits, loss
            if args.train_phase in ["finetune", "onlinelearn"]:
                mlp_error = 0
                for p, yval in y.items():
                    y_hat = model.property_mlps[p](z_prop)
                    mlp_error += mse_loss(y_hat, yval, reduction='mean')
                loss += mlp_error * args.prop_coefficient

        total_nll_abp += loss.item()
        test_data_size += x.shape[0]

    start_time = time.time()
    prior_prop_valid, uniqueness, novelty, img = eval_validity(args, test_loader, model, nbatch=args.batch_size, reconstruction=False, output_dir=output_dir, epoch=n_iter)
    prior_validity_check_time = time.time() - start_time
    logger.info('prior sample validity check time {:6.2f}'.format(prior_validity_check_time))

    rec_abp = total_nll_abp / test_data_size

    log_dict_as_table(logger, 
                     {'ABP REC': rec_abp, 
                      'Prior Validity': prior_prop_valid, 
                      'Prior Uniqueness': uniqueness, 
                      'Prior Novelty': novelty}
                     )

    model.train()
    return [prior_prop_valid, uniqueness, novelty, rec_abp, img]

if __name__ == '__main__':
    args = parse_args()
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id, args.fs_prefix, args.extra_tag)
    set_gpu(args.gpu)
    # The prior model, p_α(z), of LPT is a one-dimensional UNet where z contains 4 tokens, 
    # each of size 256. The sequence generation model, p_β(x|z), is implemented as a 
    # 3-layer causal Transformer, while a 3-layer MLP serves as the predictor model, p_γ(y|z). 
    # We perform up to 25 iterations of online learning, generating 2,500 samples per iteration,
    # which totals a maximum of 62.5K oracle function queries. We use the AdamW optimizer (Loshchilov
    # and Hutter, 2019; Kingma and Ba, 2014) with a weight decay of 0.1. Training was conducted on an
    # NVIDIA A6000 GPU, requiring 20 hours for pre-training, 10 hours for fine-tuning, and 12 hours for
    # online learning. Additional details can be found in App. A.3.
    ## train_phase == "pretrain": num_epochs=30, lr=7.5e-4, min_lr=7.5e-5
    ## train_phase == "finetune": num_epochs=10, lr=3e-4,   min_lr=7.5e-5
    ## train_phase == "onlinelearn": 
    if args.train_phase == "onlinelearn":
        run_online_learn(args, output_dir)
    else:
        run_train(args, output_dir)
