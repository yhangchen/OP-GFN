'''
    GFP
    Transformer Proxy
    Start from scratch
'''

import copy, pickle, functools
import numpy as np
import pandas as pd
import torch
from polyleven import levenshtein

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqinsertmdp, seqarmdp
from gflownet.monitor import TargetRewardDistribution, Monitor

from design_bench.datasets.discrete.gfp_dataset import GFPDataset
from design_bench.oracles.tensorflow import TransformerOracle

def dynamic_inherit_mdp(base, args):

  class GFPMDP(base):
    def __init__(self, args):
      super().__init__(args,
                       alphabet=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                                 "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                       forced_stop_len=237)
      self.args = args
      
      dataset = GFPDataset()
      self.proxy_model = TransformerOracle(dataset, noise_std=0.1)
      
      with open("datasets/gfp/rewards.pkl", "rb") as f:
        self.rewards = pickle.load(f)
        
      # scale rewards
      py = np.array(list(self.rewards))

      self.SCALE_REWARD_MAX = 10
      self.SCALE_MIN = 1e-3
      self.REWARD_EXP = 3

      py = np.maximum(py, self.SCALE_MIN)
      py = py ** self.REWARD_EXP
      self.scale = self.SCALE_REWARD_MAX / max(py)
      py = py * self.scale

      self.scaled_rewards = py

      # define modes as top % of xhashes.
      mode_percentile = 0.001
      self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))

    # Core
    @functools.lru_cache(maxsize=None)
    def reward(self, x):
      assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      # return self.scaled_oracle[x]
      pred = self.proxy_model.params["model"].predict(
        {"input_ids": np.array([self.char_to_idx[c] for c in list(x.content)]).reshape(1, -1)}
      )[0].item()
      
      r = np.maximum(pred, self.SCALE_MIN)
      r = r ** self.REWARD_EXP
      r = r * self.scale
      return r

    def is_mode(self, x, r):
      return r >= self.mode_r_threshold

    '''
      Interpretation & visualization
    '''
    def dist_func(self, state1, state2):
      """ States are SeqPAState or SeqInsertState objects. """
      return levenshtein(state1.content, state2.content)

    def make_monitor(self):
      target = TargetRewardDistribution()
      target.init_from_base_rewards(self.scaled_rewards)
      return Monitor(self.args, target, dist_func=self.dist_func,
                     is_mode_f=self.is_mode, callback=self.add_monitor)

    def add_monitor(self, xs, rs, allXtoR):
      """ Reimplement scoring with oracle, not unscaled oracle (used as R). """
      tolog = dict()
      return tolog
    
    def reduce_storage(self):
      del self.rewards
      del self.scaled_rewards

  return GFPMDP(args)


def main(args):
  print('Running experiment GFP ...')

  if args.mdp_style == 'pa':
    base = seqpamdp.SeqPrependAppendMDP
    actorclass = seqpamdp.SeqPAActor
  elif args.mdp_style == 'insert':
    base = seqinsertmdp.SeqInsertMDP
    actorclass = seqinsertmdp.SeqInsertActor
  elif args.mdp_style == 'autoregressive':
    base = seqarmdp.SeqAutoregressiveMDP
    actorclass = seqarmdp.SeqARActor
  mdp = dynamic_inherit_mdp(base, args)

  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  mdp.reduce_storage()

  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return

def eval(args):
  print('Running evaluation GFP ...')
  
  if args.mdp_style == 'pa':
    base = seqpamdp.SeqPrependAppendMDP
    actorclass = seqpamdp.SeqPAActor
  elif args.mdp_style == 'insert':
    base = seqinsertmdp.SeqInsertMDP
    actorclass = seqinsertmdp.SeqInsertActor
  elif args.mdp_style == 'autoregressive':
    base = seqarmdp.SeqAutoregressiveMDP
    actorclass = seqarmdp.SeqARActor
  mdp = dynamic_inherit_mdp(base, args)

  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  # Save memory, after constructing monitor with target rewards
  del mdp.rs_all

  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + 'final.pth')
  else:
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + f'round_{args.ckpt}.pth')
    
  # evaluate
  with torch.no_grad():
    eval_samples = model.batch_fwd_sample(args.eval_num_samples, epsilon=0.0)
    
  allXtoR = dict()
  for exp in eval_samples:
    if exp.x not in allXtoR:
      allXtoR[exp.x] = exp.r 
  
  round_num = 1
  monitor.log_samples(round_num, eval_samples)
  log = monitor.eval_samplelog(model, round_num, allXtoR)

  # save results
  result_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    result_path += '/' + 'final.pkl'
  else:
    result_path += '/' + f'round_{args.ckpt}.pkl'
    
  with open(result_path, "wb") as f:
    pickle.dump(log, f)
