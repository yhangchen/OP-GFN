from abc import ABC, abstractmethod
from copy import deepcopy
from typing import *

import torch
from gymnasium.spaces import Discrete, Space
from torchtyping import TensorType

from gfn.containers.states import States, correct_cast
from gfn.envs.preprocessors import IdentityPreprocessor, Preprocessor

from nats_bench import create

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]
PmfTensor = TensorType["n_states", torch.float]

NonValidActionsError = type("NonValidActionsError", (ValueError,), {})

from gfn.envs import Env

OPS = ["|none",
       "|skip_connect",
       "|nor_conv_1x1",
       "|nor_conv_3x3",
       "|avg_pool_3x3"]

CON = ['~0|+', '~0', '~1|+', '~0', '~1', '~2|']


class NAS(Env):
    def __init__(
            self, 
            ndim: int,
            nop: int,
            dataset: str = 'cifar10',
            device_str: Literal["cpu", "cuda"] = "cpu",
            preprocessor: Optional[Preprocessor] = None,
            log_reward_function:Callable[[StatesTensor], BatchTensor] = None,
            hp: str = '200',
            beta: float = 1.0,
        ):
        """NAS environment.

        Args:
            ndim (int, optional): number of blocks.
            nop (int, optional): number of operation for each block.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
        """
        self.ndim = ndim
        self.nop = nop
        self._api = create(None, 'tss', fast_mode=True, verbose=False)
        self.dataset = dataset
        self.log_reward_function = log_reward_function
        self.hp = hp
        self.beta = beta

        action_space = Discrete(ndim * nop + 1)
        # action i*ndim+j means setting j-th block to operation i
        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), nop, dtype=torch.long, device=torch.device(device_str))
        
        super().__init__(action_space=action_space, s0=s0, sf=sf, device_str=device_str, preprocessor=preprocessor)

    @classmethod
    def state2str(cls, final_state: OneStateTensor) -> str:
        return ''.join([OPS[j]+CON[i] for i, j in enumerate(final_state)])
    
    def make_States_class(self) -> type[States]:
        env = self

        class DAGStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            # sampling random terminating states tensor
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                return torch.randint(
                    0, env.nop, batch_shape + (env.ndim,), device=env.device
                )
            
            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                forward_masks = torch.zeros(
                    (*self.batch_shape, env.n_actions),
                    dtype=torch.bool,
                    device=env.device,
                )
                backward_masks = torch.zeros(
                    (*self.batch_shape, env.n_actions - 1),
                    dtype=torch.bool,
                    device=env.device,
                )

                return forward_masks, backward_masks

                
            def update_masks(self) -> None:
                "Update the masks based on the current states."
                # The following two lines are for typing only.
                self.forward_masks = cast(ForwardMasksTensor, self.forward_masks)
                self.backward_masks = cast(BackwardMasksTensor, self.backward_masks)

                for i in range(env.nop):
                    self.forward_masks[..., i*env.ndim:(i+1)*env.ndim] = self.states_tensor == -1
                    self.backward_masks[..., i*env.ndim:(i+1)*env.ndim] = self.states_tensor == i
                self.forward_masks[..., -1] = torch.all(self.states_tensor != -1, dim=-1)

        return DAGStates

    def is_exit_actions(self, actions: BatchTensor) -> BatchTensor:
        return actions == self.n_actions - 1
    
    def maskless_step(self, states: StatesTensor, actions: BatchTensor) -> None:
        for i in range(self.nop):
            mask_i = (actions >= i * self.ndim) & (actions < (i + 1) * self.ndim)
            states[mask_i] = states[mask_i].scatter_(-1, (actions[mask_i] - i * self.ndim).unsqueeze(-1), i)
        mask_f = actions == self.ndim * self.nop + 1
        states[mask_f] = self.sf.repeat(mask_f.sum(),1)

    def maskless_backward_step(
        self, states: StatesTensor, actions: BatchTensor
    ) -> None:
        states.scatter_(-1, actions.unsqueeze(-1).fmod(self.ndim), -1)

    def is_final(self, states: StatesTensor) -> BatchTensor:
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        return torch.all(states != -1, dim=-1)

    def accuracy(self, final_state: OneStateTensor) -> float:
        if final_state.nelement() == 0: # deal with empty tensor
            return 0.0
        if self.is_final(final_state):
            if len(final_state.shape) == 2:
                final_state = final_state.squeeze()
            return self._api.get_more_info(self._api.archstr2index[self.state2str(final_state)], 
                                           self.dataset, hp=self.hp, is_random=False)['test-accuracy']
        else:
            return 0.0
        
    def log_reward(self, states: States) -> BatchTensor:
        states_raw = states.states_tensor
        if self.log_reward_function is not None:
            if states_raw.shape[0] == 0:
                return torch.tensor([], dtype=torch.float, device=self.device)
            return self.log_reward_function(states_raw).to(self.device)
        
        # else use default reward, i.e. accuracy.
        acc = torch.tensor(tuple(map(self.accuracy, states_raw.split(1))), device=self.device)
        return torch.log(acc/100) * self.beta

    @property
    def n_states(self) -> int:
        return self.ndim**(self.nop+1)
    
    @property
    def n_terminating_states(self) -> int:
        return self.ndim**self.nop
