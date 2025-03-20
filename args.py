import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true')
    parser.add_argument('--wandb', action='store_true')

    # Input data
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--mol_property', default=[], type=lambda arg: [a.strip() for a in arg.split(',')])
    parser.add_argument('--mask_nonnegative', action='store_true')
    parser.add_argument('--train_from', default='')
    parser.add_argument('--max_len', default=73, type=int)
    parser.add_argument('--batch_size', default=896, type=int)
    parser.add_argument('--extra_tag', type=str, default="")
    parser.add_argument('--autodock_executable', type=str, default='AutoDock-GPU/bin/autodock_gpu_128wi')
    parser.add_argument('--protein_file', type=str, default='data/1err/1err.maps.fld')
    parser.add_argument('--cond_substructure_smiles', type=str, default=None)
    parser.add_argument('--num_autodock_proc', type=int, default=32)

    # SRI options
    parser.add_argument('--z_n_iters', type=int, default=15)
    parser.add_argument('--z_step_size', type=float, default=0.3)
    parser.add_argument('--z_with_noise', default=True, type=bool)
    parser.add_argument('--noise_factor', type=float, default=0.5)

    # Model options
    parser.add_argument('--latent_dim', default=256, type=int)
    parser.add_argument('--mlp_hidden_dim', default=100, type=int)
    parser.add_argument('--mlp_dropout_rate', default=0.3, type=float)
    parser.add_argument('--n_latent', default=8, type=int) 
    parser.add_argument('--dec_word_dim', default=512, type=int)
    parser.add_argument('--prop_coefficient', default=1., type=float)
    parser.add_argument('--single_design', action='store_true')
    parser.add_argument('--train_phase', default='pretrain', type=str, choices=['pretrain', 'finetune', 'onlinelearn'])

    # Optimization options
    parser.add_argument('--fs_prefix', default='./', type=str)
    parser.add_argument('--checkpoint_dir', default='models', type=str)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--lr', default=7.5e-4, type=float)
    parser.add_argument('--min_lr', default=7.5e-5, type=float)
    parser.add_argument('--warmup_iters', default=600, type=int)
    parser.add_argument('--max_lr_iters', default=2000, type=int)
    parser.add_argument('--lr_decay_iters', default=9000, type=int)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=3435, type=int)
    parser.add_argument('--print_every', type=int, default=1)

    return parser.parse_args()
