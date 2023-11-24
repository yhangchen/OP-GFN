"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

from dataclasses import dataclass
import copy

import torch
from torchtyping import TensorType
from gfn.containers.states import correct_cast
from gfn.containers import Trajectories
from gfn.estimators import LogZEstimator
from gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss, ordering_loss
from gfn.samplers.actions_samplers import (
    BackwardDiscreteActionsSampler,
    DiscreteActionsSampler,
)

# Typing
ScoresTensor = TensorType["n_trajectories", float]
LossTensor = TensorType[0, float]


@dataclass
class TBParametrization(PFBasedParametrization):
    r"""
    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    $\mathcal{O}_3$ is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if self.logit_PB is a fixed LogitPBEstimator.
    Useful for the Trajectory Balance Loss.
    """
    logZ: LogZEstimator


class TrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: TBParametrization,
        log_reward_clip_min: float = -12,
        on_policy: bool = False,
    ):
        """Loss object to evaluate the TB loss on a batch of trajectories.

        Args:
            log_reward_clip_min (float, optional): minimal value to clamp the reward to. Defaults to -12 (roughly log(1e-5)).
            on_policy (bool, optional): If True, the log probs stored in the trajectories are used. Which should be faster than
                                        reevaluating them. Defaults to False.
        """
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )
        self.on_policy = on_policy

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores + self.parametrization.logZ.tensor).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

class KLTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: TBParametrization,
        log_reward_clip_min: float = -12,
        on_policy: bool = False,
        KL_weight: float = 0.0,
        train_backward: bool = True,
    ):
        """Loss object to evaluate the TB loss on a batch of trajectories.

        Args:
            log_reward_clip_min (float, optional): minimal value to clamp the reward to. Defaults to -12 (roughly log(1e-5)).
            on_policy (bool, optional): If True, the log probs stored in the trajectories are used. Which should be faster than
                                        reevaluating them. Defaults to False.
        """
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        
        if not train_backward:
            parametrization.logit_PB.module = copy.deepcopy(parametrization.logit_PB.module)
            for p in parametrization.logit_PB.module.parameters():
                p.requires_grad = False
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )
        self.on_policy = on_policy
        self.KL_weight = KL_weight

    def get_backward_entropy(self, trajectories: Trajectories) -> ScoresTensor:
        if trajectories.is_backward:
            raise ValueError("Backward trajectories are not supported")

        valid_states = trajectories.states[~trajectories.states.is_sink_state]
        valid_actions = trajectories.actions[trajectories.actions != -1]

        if valid_states.batch_shape != tuple(valid_actions.shape):
            raise AssertionError("Something wrong happening with log_pf evaluations")

        valid_states.forward_masks, valid_states.backward_masks = correct_cast(
            valid_states.forward_masks, valid_states.backward_masks
        )

        valid_states = valid_states[~valid_states.is_initial_state]

        valid_pb_logits = self.backward_actions_sampler.get_logits(valid_states)
        valid_pb_all = valid_pb_logits.softmax(dim=-1)
        uniform_dist = (
                        valid_states.backward_masks.float()
                        / valid_states.backward_masks.sum(dim=-1, keepdim=True).float()
                    )
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

        eps = torch.finfo(valid_pb_all.dtype).eps

        valid_pb_all = valid_pb_all.clamp(min=eps, max=1 - eps)
        
        if len(valid_pb_all)==0:
            return 0.0
        return kl_loss(valid_pb_all.log(), uniform_dist)
    


    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores + self.parametrization.logZ.tensor).pow(2).mean() \
            + self.KL_weight * self.get_backward_entropy(trajectories=trajectories)
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


class orderingedTrajectoryBalance(KLTrajectoryBalance):
    def __init__(
        self,
        parametrization: TBParametrization,
        log_reward_clip_min: float = -12,
        on_policy: bool = False,
        KL_weight: float = 0.0,
        CE_weight: float = 0.0,
        train_backward: bool = True,
    ):
        self.CE_weight = CE_weight
        
        super().__init__(
            parametrization=parametrization,
            log_reward_clip_min=log_reward_clip_min,
            on_policy=on_policy,
            KL_weight=KL_weight,
            train_backward=train_backward,
        )
    


    def __call__(self, trajectories: Trajectories) -> LossTensor:
        log_pf, log_pb, scores = self.get_trajectories_scores(trajectories)
        log_rewards = log_pf - log_pb - scores
        pred_log_rewards = log_pf - log_pb + self.parametrization.logZ.tensor
        loss = self.KL_weight * self.get_backward_entropy(trajectories=trajectories) \
            + ordering_loss(log_rewards, pred_log_rewards)
        if torch.isnan(loss):
            raise ValueError("loss is nan")
        return loss


class LogPartitionVarianceLoss(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: PFBasedParametrization,
        log_reward_clip_min: float = -12,
        on_policy: bool = False,
    ):
        """Loss object to evaluate the Log Partition Variance Loss (Section 3.2 of
        [ROBUST SCHEDULING WITH GFLOWNETS](https://arxiv.org/abs/2302.05446))

        Args:
            on_policy (bool, optional): If True, the log probs stored in the trajectories are used. Which should be faster than
                                        reevaluating them. Defaults to False.
        """
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )

        self.on_policy = on_policy

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores - scores.mean()).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss
