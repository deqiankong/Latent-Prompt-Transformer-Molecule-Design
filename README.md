# Molecule Design by Latent Prompt Transformer

This repository contains code for NeurIPS 2024 spotlight paper [Molecule Design by Latent Prompt Transformer](https://arxiv.org/abs/2402.17179).

```bash
# Install uv if necessary
which uv >/dev/null 2>&1 || curl -sSf https://install.astral.sh/uv | sh
# Setup Python environment
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
# Download Open Babel
curl -L https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1-source.tar.bz2 | tar -xj
## # Download AutoDock-GPU for your system
## curl -O https://github.com/ccsb-scripps/AutoDock-GPU/releases/download/v1.6/adgpu-v1.6_linux_x64_cuda12_128wi
# Compile AutoDock-GPU for your system
git clone https://github.com/ccsb-scripps/AutoDock-GPU.git
cd AutoDock-GPU
export GPU_INCLUDE_PATH=/usr/include
export GPU_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
make DEVICE=CUDA
```

## Usage

There are three training phases: `pretrain`, `finetune`, `onlinelearn`.

The expected data format can be seen in `utils.py`. Tokenized SELFIES strings should be in `data/Xsf{train,test}.npy`; matching unpadded lengths should be in `data/Lsf{train,test}.npy`. Similarly, matching molecule sequence property values should be in `data/$MP_{train,test}.npy`, where `$MP` denotes the property of interest. We support: `['ba0', 'ba1', 'phgdh', 'PlogP', 'sas', 'qed']`, noting `ba0=ESR1` `ba1=acaa1` `phgdh=ba2`.

For checking novelty, uniqueness, and validity, we provide a random sample of ZINC in `data/test_5.txt`.

### `pretrain`

In this phase we learn $p(x|z)$, ignoring target properties $y$.

```bash
python train.py --train_phase=pretrain --num_epochs=30 --lr=7.5e-4 --min_lr=7.5e-5 --gpu=0
```

### `finetune`

In this phase, which can be considered part of pretraining, we learn $p(y|z)$ jointly with the initially pretrained $p(x|z)$.

Make sure to specify a model checkpoint from the `pretrain` phase as `--train_from`.

Example using ba0:

```bash
python train.py --train_phase=finetune --num_epochs=10 --lr=3e-4 --min_lr=7.5e-5 --mol_property=ba0 --train_from=$PRETRAIN_CHECKPOINT --gpu=0
```

### `onlinelearn`

In this phase we shift the distribution of the learned generative model.

Make sure to specify a protein file; we provide ba0: `1err.maps.fld`, ba1: `2iik.maps.fld`, phgdh: `PHGDH2_NAD_BindingDomain.maps.fld`.

Make sure to specify the property of interest and specify a model checkpoint from the `finetune` phase as `--train_from`.

Example using ba0:

```bash
python train.py --train_phase=onlinelearn --num_epochs=25 --lr=5e-4 --min_lr=7.5e-5 --mol_property=ba0 --protein_file=data/1err.maps.fld --train_from=$FINETUNE_CHECKPOINT --gpu=0
```

## Questions

Please open an issue or email e.honig@ucla.edu with any questions or comments regarding this code.

Thank you for your interest in our work!

## Cite

```
@inproceedings{kong2024molecule,
 title = {Molecule Design by Latent Prompt Transformer},
 author = {Kong, Deqian and Huang, Yuhao and Xie, Jianwen and Honig, Edouardo and Xu, Ming and Xue, Shuanghong and Lin, Pei and Zhou, Sanping and Zhong, Sheng and Zheng, Nanning and Wu, Ying Nian},
 booktitle = {Advances in Neural Information Processing Systems},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/a229cb89a98a84b2373496bb3cfc3570-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
