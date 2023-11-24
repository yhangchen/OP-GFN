from typing import List

import numpy as np
import torch

from gflownet.config import Config
from botorch.utils.multi_objective import pareto


class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = []
        self.position = 0
        self.rng = rng

    def push(self, *args):
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = self.rng.choice(len(self.buffer), batch_size)
        out = list(zip(*[self.buffer[idx] for idx in idxs]))
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out)

    def __len__(self):
        return len(self.buffer)
    
class MOOReplayBuffer(ReplayBuffer):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        super().__init__(cfg=cfg, rng=rng)
        self.pareto_indices = []

    def push(self, *args):
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.update_pareto(args)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity
        
    def update_pareto(self, new):
        flat_rewards = np.stack([self.buffer[idx][2] for idx in self.pareto_indices] + [new[2]], axis=0)
        inds = np.array(pareto.is_non_dominated(torch.tensor(flat_rewards), deduplicate=False))
        candidate_indices = np.array(self.pareto_indices + [self.position])
        self.pareto_indices = candidate_indices[inds].tolist()

        
    def sample_pareto(self, batch_size):
        # num_pareto = min(batch_size, len(self.pareto_indices))
        idxs_1 = self.rng.choice(self.pareto_indices, batch_size)
        out = list(zip(*[self.buffer[idx] for idx in idxs_1]))
        for i in range(len(out)):
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out)