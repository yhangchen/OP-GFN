from typing import List, Optional

import torch
from torchtyping import TensorType

from gfn.containers import States, Trajectories
from gfn.envs import Env
from gfn.samplers.actions_samplers import ActionsSampler, BackwardActionsSampler, ParallelDiscreteActionsSampler

# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
LogProbsTensor = TensorType["n_trajectories", torch.float]
DonesTensor = TensorType["n_trajectories", torch.bool]


class TrajectoriesSampler:
    def __init__(
        self,
        env: Env,
        actions_sampler: ActionsSampler,
    ):
        """Sample complete trajectories, or completes trajectories from a given batch states, using actions_sampler.

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.is_backward = isinstance(actions_sampler, BackwardActionsSampler)

    def sample_trajectories(
        self,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
        n_trajectories_per_terminal: Optional[int] = None,
    ) -> Trajectories:
        if states is None:
            assert (
                n_trajectories is not None
            ), "Either states or n_trajectories should be specified"
            states = self.env.reset(batch_shape=(n_trajectories,), random=self.is_backward)
            if self.is_backward:
                while all(states.is_initial_state):
                    states = self.env.reset(batch_shape=(n_trajectories,), random=self.is_backward)
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        if not self.is_backward or n_trajectories_per_terminal is None:
            n_trajectories_per_terminal = 1
        states = self.env.States(states.states_tensor.repeat(n_trajectories_per_terminal, 1))
        n_trajectories *= n_trajectories_per_terminal

        device = states.states_tensor.device

        dones = states.is_initial_state if self.is_backward else states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states_tensor]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_logprobs: List[LogProbsTensor] = []
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0

        while not all(dones):
            actions = torch.full(
                (n_trajectories,),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )
            actions_log_probs, valid_actions = self.actions_sampler.sample(
                states[~dones]
            )
            actions[~dones] = valid_actions.to(device)
            log_probs[~dones] = actions_log_probs.to(device)
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]

            if self.is_backward:
                new_states = self.env.backward_step(states, actions)
            else:
                new_states = self.env.step(states, actions)
            sink_states_mask = new_states.is_sink_state

            step += 1

            new_dones = (
                new_states.is_initial_state if self.is_backward else sink_states_mask
            ) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            try:
                trajectories_log_rewards[new_dones & ~dones] = self.env.log_reward(
                    states[new_dones & ~dones]
                )
            except NotImplementedError:
                # print(states[new_dones & ~dones])
                # print(torch.log(self.env.reward(states[new_dones & ~dones])))
                trajectories_log_rewards[new_dones & ~dones] = torch.log(
                    self.env.reward(states[new_dones & ~dones])
                )
            states = new_states
            dones = dones | new_dones

            trajectories_states += [states.states_tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = self.env.States(states_tensor=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0)
        trajectories_logprobs = torch.stack(trajectories_logprobs, dim=0)

        trajectories = Trajectories(
            env=self.env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.is_backward,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,
        )

        return trajectories

    def sample(self, n_trajectories: int, n_trajectories_per_terminal: Optional[int] = None) -> Trajectories:
        return self.sample_trajectories(n_trajectories=n_trajectories, n_trajectories_per_terminal = n_trajectories_per_terminal)

class CorrectedTrajectoriesSampler:
    def __init__(
        self,
        env: Env,
        actions_sampler: ActionsSampler,
        backward_actions_sampler: BackwardActionsSampler,
    ):
        """

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.backward_actions_sampler = backward_actions_sampler

    def sample_trajectories(
        self,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
    ) -> Trajectories:
        if states is None:
            assert (
                n_trajectories is not None
            ), "Either states or n_trajectories should be specified"
            states = self.env.reset(batch_shape=(n_trajectories,), random=False)
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        device = states.states_tensor.device

        dones = states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states_tensor]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_logprobs: List[LogProbsTensor] = []
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0

        while not all(dones):
            actions = torch.full((n_trajectories,), fill_value=-1, dtype=torch.long, device=device)
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )

            # predict
            actions_log_probs, valid_actions = self.actions_sampler.sample(
                states[~dones]
            )
            actions[~dones] = valid_actions.to(device)
            log_probs[~dones] = actions_log_probs.to(device)
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]

            new_states = self.env.step(states, actions)                
            sink_states_mask = new_states.is_sink_state
            init_states_mask = new_states.is_initial_state
            new_dones = sink_states_mask & ~dones

            new_dones = dones | new_dones
            

            # correct forward
            # non_init_dones = ~new_dones & ~init_states_mask
            _, corrector_valid_actions = self.actions_sampler.sample(
                new_states[~new_dones]
            )
            
            actions = torch.full((n_trajectories,), fill_value=-1, dtype=torch.long, device=device)
            actions[~new_dones] = corrector_valid_actions
            new_states_0 = self.env.step(new_states, actions)
            new_states.states_tensor[~new_states_0.is_sink_state] = new_states_0.states_tensor[~new_states_0.is_sink_state]
            new_states = self.env.States(new_states.states_tensor)
     
            # correct backward
            non_init_sink = ~new_states_0.is_sink_state & ~new_states_0.is_initial_state
            _, corrector_backward_valid_actions = self.backward_actions_sampler.sample(
                new_states[non_init_sink]
            )            
            new_states_1 = self.env.backward_step(new_states[non_init_sink], corrector_backward_valid_actions)
            new_states.states_tensor[non_init_sink] = new_states_1.states_tensor
            new_states = self.env.States(new_states.states_tensor)

            sink_states_mask = new_states.is_sink_state
            new_dones = sink_states_mask & ~dones

            step += 1
            trajectories_dones[new_dones & ~dones] = step

            try:
                trajectories_log_rewards[new_dones & ~dones] = self.env.log_reward(
                    states[new_dones & ~dones]
                )
            except NotImplementedError:
                # print(states[new_dones & ~dones])
                # print(torch.log(self.env.reward(states[new_dones & ~dones])))
                trajectories_log_rewards[new_dones & ~dones] = torch.log(
                    self.env.reward(states[new_dones & ~dones])
                )
            states = new_states
            dones = dones | new_dones
            
            trajectories_states += [states.states_tensor]

            

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = self.env.States(states_tensor=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0)
        trajectories_logprobs = torch.stack(trajectories_logprobs, dim=0)

        trajectories = Trajectories(
            env=self.env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=False,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,
        )

        return trajectories

    def sample(self, n_trajectories: int) -> Trajectories:
        return self.sample_trajectories(n_trajectories=n_trajectories)

import copy
class MHCorrectedTrajectoriesSampler:
    def __init__(
        self,
        env: Env,
        actions_sampler: ActionsSampler,
        backward_actions_sampler: BackwardActionsSampler,
        parametrization,
    ):
        """

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.backward_actions_sampler = backward_actions_sampler
        self.parametrization = parametrization

    def sample_terminal(
        self,
        states: Optional[States] = None,
        n_trajectories: Optional[int] = None,
        stepsize: int = 1,
    ) -> States:
        if states is None:
            assert (
                n_trajectories is not None
            ), "Either states or n_trajectories should be specified"
            states = self.env.reset(batch_shape=(n_trajectories,), random=False)
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        device = states.states_tensor.device

        dones = states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states_tensor]
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0

        while not all(dones):
            actions = torch.full((n_trajectories,), fill_value=-1, dtype=torch.long, device=device)

            # predict
            if isinstance(self.actions_sampler, ParallelDiscreteActionsSampler):
                new_states = self.actions_sampler.parallel_sample(states, stepsize=stepsize)
            else:
                _, valid_actions = self.actions_sampler.sample(
                    states[~dones]
                )
                actions[~dones] = valid_actions.to(device)

                new_states = self.env.step(states, actions)
            
            sink_states_mask = new_states.is_sink_state
            new_dones = sink_states_mask & ~dones

            new_dones = dones | new_dones
            

            # correct forward
            # non_init_dones = ~new_dones & ~init_states_mask
            corrector_valid_log_probs, corrector_valid_actions = self.actions_sampler.sample(
                new_states[~new_dones]
            )
            
            saved_states = copy.deepcopy(new_states)
            
            actions = torch.full((n_trajectories,), fill_value=-1, dtype=torch.long, device=device)
            actions[~new_dones] = corrector_valid_actions
            new_new_states = self.env.step(new_states, actions)
     
            # correct backward
            _, corrector_backward_valid_actions = self.backward_actions_sampler.sample(
                new_new_states[~new_new_states.is_sink_state]
            )            
            new_states_choice = self.env.backward_step(new_new_states[~new_new_states.is_sink_state], corrector_backward_valid_actions)
            new_states.states_tensor[~new_new_states.is_sink_state] = new_states_choice.states_tensor
            new_states = self.env.States(new_states.states_tensor)
            

            saved = saved_states[~new_new_states.is_sink_state]
            new_new = new_new_states[~new_new_states.is_sink_state]
            prop = new_states[~new_new_states.is_sink_state]

            mh_ratio_log = self.parametrization.logF(prop) - self.parametrization.logF(saved) \
                + get_log_probs(new_new, actions[~new_new_states.is_sink_state], self.backward_actions_sampler) \
                    - get_log_probs(saved, actions[~new_new_states.is_sink_state], self.actions_sampler) \
                        + get_log_probs(prop, corrector_backward_valid_actions, self.actions_sampler) \
                            - get_log_probs(new_new, corrector_backward_valid_actions, self.backward_actions_sampler)
                            
            mh_ratio = torch.exp(mh_ratio_log)
            threshold = torch.rand((len(new_new_states[~new_new_states.is_sink_state]),1), device=device)
            not_change = mh_ratio < threshold
            not_change = torch.nonzero(~new_new_states.is_sink_state, as_tuple=True)[0][not_change.squeeze()]
            
            if len(not_change):
                # print(f'not change {len(not_change)}')
                new_states.states_tensor[not_change] = saved_states.states_tensor[not_change]
                new_states = self.env.States(new_states.states_tensor)

            sink_states_mask = new_states.is_sink_state
            new_dones = sink_states_mask & ~dones

            step += 1
            trajectories_dones[new_dones & ~dones] = step

            try:
                trajectories_log_rewards[new_dones & ~dones] = self.env.log_reward(
                    states[new_dones & ~dones]
                )
            except NotImplementedError:
                # print(states[new_dones & ~dones])
                # print(torch.log(self.env.reward(states[new_dones & ~dones])))
                trajectories_log_rewards[new_dones & ~dones] = torch.log(
                    self.env.reward(states[new_dones & ~dones])
                )
            states = new_states
            dones = dones | new_dones
            
            
            trajectories_states += [states.states_tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0)
        trajectories_states = self.env.States(states_tensor=trajectories_states)
        
        return trajectories_states[trajectories_dones - 1, torch.arange(n_trajectories)]

    def sample(self, n_trajectories: int, stepsize: int = 1) -> States:
        return self.sample_terminal(n_trajectories=n_trajectories, stepsize=stepsize)


from torch.distributions import Categorical
def get_log_probs(states: States, actions, sampler):
    probs = sampler.get_probs(states).type(torch.DoubleTensor).to(states.device)
    dist = Categorical(probs=probs)
    actions_log_probs = dist.log_prob(actions).float()
    return actions_log_probs[:,None]