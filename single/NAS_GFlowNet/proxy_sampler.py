from gfn.samplers import TrajectoriesSampler, ActionsSampler
from gfn.containers import States, Trajectories
from gfn.envs import Env
from gfn.losses import TBParametrization, TrajectoryBalance
from gfn.estimators import LogEdgeFlowEstimator, LogitPBEstimator, LogitPFEstimator
import torch

class ProxyTrajectoriesSampler(TrajectoriesSampler):
    def __init__(
        self,
        env: Env,
        actions_sampler: ActionsSampler,
        parametrization: TBParametrization,
    ):
        super().__init__(
            env=env, actions_sampler=actions_sampler
        )
        self.parametrization=parametrization
        self.loss_fn = TrajectoryBalance(parametrization=self.parametrization)
    
    def proxy_sample(self, n_trajectories: int, n_candidates: int) -> Trajectories:
        assert n_trajectories <= n_candidates
        trajectories = self.sample(n_candidates)
        log_pf, log_pb, _ = self.loss_fn.get_trajectories_scores(trajectories)
        pred_log_rewards = log_pf - log_pb + self.parametrization.logZ.tensor
        indices = torch.topk(pred_log_rewards, n_trajectories).indices
        return trajectories[indices]