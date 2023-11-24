from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from gfn.containers.states import States, correct_cast
from gfn.estimators import LogEdgeFlowEstimator, LogitPBEstimator, LogitPFEstimator

import numpy as np

# Typing
Tensor2D = TensorType["batch_size", "n_actions"]
Tensor2D2 = TensorType["batch_size", "n_steps"]
Tensor1D = TensorType["batch_size", torch.long]


class ActionsSampler(ABC):
    """
    Base class for action sampling methods.
    """

    @abstractmethod
    def sample(self, states: States) -> Tuple[Tensor1D, Tensor1D]:
        """
        Args:
            states (States): A batch of states.

        Returns:
            Tuple[Tensor[batch_size], Tensor[batch_size]]: A tuple of tensors containing the log probabilities of the sampled actions, and the sampled actions.
        """
        pass


class BackwardActionsSampler(ActionsSampler):
    """
    Base class for backward action sampling methods.
    """

    pass


class DiscreteActionsSampler(ActionsSampler):
    """
    For Discrete environments.
    """

    def __init__(
        self,
        estimator: LogitPFEstimator | LogEdgeFlowEstimator,
        temperature: float = 1.0,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
    ) -> None:
        """Implements a method that samples actions from any given batch of states.

        Args:
            temperature (float, optional): scalar to divide the logits by before softmax. Defaults to 1.0.
            sf_bias (float, optional): scalar to subtract from the exit action logit before dividing by temperature. Defaults to 0.0.
            epsilon (float, optional): with probability epsilon, a random action is chosen. Defaults to 0.0.
        """
        self.estimator = estimator
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon

    def get_raw_logits(self, states: States) -> Tensor2D:
        """
        This is before illegal actions are masked out and the exit action is biased.
        Should be used for Discrete action spaces only.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the logits for each action in each state in the batch.
        """
        logits = self.estimator(states)
        return logits

    def get_logits(self, states: States) -> Tensor2D:
        """Transforms the raw logits by masking illegal actions.

        Raises:
            ValueError: if one of the resulting logits is NaN.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the transformed logits.
        """
        logits = self.get_raw_logits(states)

        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        states.forward_masks, _ = correct_cast(
            states.forward_masks, states.backward_masks
        )
        logits[~states.forward_masks] = -float("inf")
        return logits

    def get_probs(
        self,
        states: States,
    ) -> Tensor2D:
        """
        Returns:
            The probabilities of each action in each state in the batch.
        """
        logits = self.get_logits(states)
        logits[..., -1] -= self.sf_bias
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return probs

    def sample(self, states: States) -> Tuple[Tensor1D, Tensor1D]:
        device = states.states_tensor.device
        probs = self.get_probs(states).type(torch.DoubleTensor).to(device)
        # torch.multinomial might select zero probability action, we cast it to float64 to make this happen in less probability
        states.forward_masks, _ = correct_cast(
            states.forward_masks, states.backward_masks
        )
        if self.epsilon > 0:
            uniform_dist = (
                states.forward_masks.float()
                / states.forward_masks.sum(dim=-1, keepdim=True).float()
            )
            probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist
        dist = Categorical(probs=probs)
        with torch.no_grad():
            actions = dist.sample()
        actions_log_probs = dist.log_prob(actions).float()

        return actions_log_probs, actions


class BackwardDiscreteActionsSampler(DiscreteActionsSampler, BackwardActionsSampler):
    """
    For sampling backward actions in discrete environments.
    """

    def __init__(
        self,
        estimator: LogitPBEstimator,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> None:
        """s_f is not biased in the backward sampler."""
        super().__init__(
            estimator, temperature=temperature, sf_bias=0.0, epsilon=epsilon
        )

    def get_logits(self, states: States) -> Tensor2D:
        logits = self.get_raw_logits(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        _, states.backward_masks = correct_cast(
            states.forward_masks, states.backward_masks
        )
        logits[~states.backward_masks] = -float("inf")
        return logits

    def get_probs(self, states: States) -> Tensor2D:
        logits = self.get_logits(states)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        # The following line is hack that works: when probs are nan, it means
        # that the state is already done (usually during backward sampling).
        # In which case, any action can be passed to the backward_step function
        # making the state stay at s_0
        probs = probs.nan_to_num(nan=1.0 / probs.shape[-1])
        return probs

    def sample(self, states: States) -> Tuple[Tensor1D, Tensor1D]:
        device = states.states_tensor.device
        probs = self.get_probs(states).type(torch.DoubleTensor).to(device)
        # torch.multinomial might select zero probability action, we cast it to float64 to make this happen in less probability
        _, states.backward_masks = correct_cast(
            states.forward_masks, states.backward_masks
        )
        if self.epsilon > 0:
            uniform_dist = (
                states.backward_masks.float()
                / states.backward_masks.sum(dim=-1, keepdim=True).float()
            )
            probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist
        dist = Categorical(probs=probs)
        with torch.no_grad():
            actions = dist.sample()
        actions_log_probs = dist.log_prob(actions).float()

        return actions_log_probs, actions


class ParallelDiscreteActionsSampler(DiscreteActionsSampler):
    def __init__(
        self,
        estimator: LogitPFEstimator | LogEdgeFlowEstimator,
        temperature: float = 1.0,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(
            estimator=estimator,
            temperature=temperature,
            sf_bias=sf_bias,
            epsilon=epsilon,
        )
    
    def parallel_sample(self, states: States, stepsize=None) -> States:
        env = self.estimator.env
        device = states.states_tensor.device
        probs = self.get_probs(states).type(torch.DoubleTensor).to(device)
        states.forward_masks, _ = correct_cast(
            states.forward_masks, states.backward_masks
        )
        if self.epsilon > 0:
            uniform_dist = (
                states.forward_masks.float()
                / states.forward_masks.sum(dim=-1, keepdim=True).float()
            )
            probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist
        
        ndim = states.states_tensor.shape[-1]
        nop = probs.shape[-1]//ndim
        
        num_filled = (states.states_tensor!=-1).sum(dim=-1)
        assert all(num_filled==num_filled[0])
        empty = ndim - int(num_filled[0])
        if stepsize is None:
            stepsize = 1 # how many step to perform

        
        probs_reshape = probs[:,:-1].reshape(-1, nop, ndim)
        probs_reshape = torch.swapaxes(probs_reshape, 1, 2)
        probs_position = probs_reshape.sum(dim=-1)
        
        dist = Categorical(probs=probs_reshape+torch.finfo(probs.dtype).eps)
        with torch.no_grad():
            all_actions = dist.sample()
        
        mask_list = []
        terminal_ind = []
        for ind, prob_position in enumerate(probs_position):
            prob_position = prob_position.detach().numpy()
            if np.sum(prob_position) < torch.finfo(probs.dtype).eps:
                terminal_ind.append(ind)
                mask = torch.zeros((1, ndim))
                mask_list.append(mask)
                continue
            prob_position /= np.sum(prob_position)
            mask_ind = np.random.choice(ndim, stepsize, replace=False, p=prob_position.tolist())
            mask = torch.zeros((1, ndim))
            mask[0,mask_ind] = 1
            mask_list.append(mask)
            
        full_mask = torch.cat(mask_list, dim=0).to(env.device)
        all_actions = torch.where(full_mask==1, all_actions, -1).to(env.device)
        new_states_tensor = torch.where(states.states_tensor==-1, all_actions, states.states_tensor)
        new_states_tensor[np.array(terminal_ind)] = env.sf[None,:].repeat(len(terminal_ind),1)
        
        return env.States(new_states_tensor)
