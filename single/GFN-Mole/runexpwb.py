'''
  Run experiment with wandb logging.

  Usage:
  python runexpwb.py --setting bag

  Note: wandb isn't compatible with running scripts in subdirs:
    e.g., python -m exps.chess.chessgfn
  So we call wandb init here.
'''
import random
import torch
import wandb
import options
import numpy as np
from attrdict import AttrDict

from exps.bag import bag
from exps.tfbind8 import tfbind8_oracle
from exps.tfbind10 import tfbind10
from exps.qm9str import qm9str
from exps.sehstr import sehstr
# from exps.gfp import gfp
# from exps.utr import utr
from exps.rna import rna

setting_calls = {
  'bag': lambda args: bag.main(args),
  'tfbind8': lambda args: tfbind8_oracle.main(args),
  'tfbind10': lambda args: tfbind10.main(args),
  'qm9str': lambda args: qm9str.main(args),
  'sehstr': lambda args: sehstr.main(args),
  # 'gfp': lambda args: gfp.main(args),
  # 'utr': lambda args: utr.main(args),
  'rna': lambda args: rna.main(args),
}


def main(args):
  print(f'Using {args.setting} ...')
  exp_f = setting_calls[args.setting]
  exp_f(args)
  return

def set_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


if __name__ == '__main__':
  args = options.parse_args()

  if args.model == 'sub':
    args.guide = 'substructure'
  
  set_seed(args.seed)

  if not args.ordering:
    wandb_name = f'{args.setting}-{args.model}'
  else:
    wandb_name = f'{args.setting}-OP{args.model}'

  
  # RNA Binding - 4 different tasks
  if args.setting == "rna":
    args.saved_models_dir = f"{args.saved_models_dir}/L{args.rna_length}_RNA{args.rna_task}/" 
    wandb.init(project=f"{args.wandb_project}-L{args.rna_length}-{args.rna_task}",
              entity=args.wandb_entity,
              config=args,
              mode=args.wandb_mode, name=wandb_name)
  else:
    wandb.init(project=args.wandb_project,
              entity=args.wandb_entity,
              config=args, 
              mode=args.wandb_mode, name=wandb_name)
  args = AttrDict(wandb.config)
  # args.run_name = wandb.run.name if wandb.run.name else 'None'
  run_name = args.model
  if args.model == 'subtb':
    run_name += f"{args.lamda}"
  
  if args.offline_select == "prt":
    run_name += "_" + args.offline_select
  
  if args.sa_or_ssr == "ssr":
    run_name += "_" + args.sa_or_ssr


  if args.mcmc == True:
    run_name += "_" + "mcmc"
    if args.mh == True:
      run_name += "_" + "mh"
    run_name += "_" + f"k{args.k}"
    run_name += "_" + f"c{args.num_chain}"
    
  run_name += "_" + f"beta{args.reward_exp}"
  run_name += "_" + f"seed{args.seed}"
  
  if args.model == "sql":
    run_name += f"/lr{args.lr_policy}_entropy{args.entropy_coef}"
  
  args.run_name = run_name.upper()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'device={device}')
  args.device = device
 
  main(args)
