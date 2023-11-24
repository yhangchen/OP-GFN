import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from gfn import *
from gfn.losses import *
from gfn.samplers import *
from gfn.containers import *
from gfn.envs.preprocessors import *
from nas import NAS
from args import get_parser
import copy
from proxy_sampler import ProxyTrajectoriesSampler

def prepare_training(env_train: NAS, env_valid: NAS, args):
    if args.loss_type == 'TB':
        logit_PF = LogitPFEstimator(env=env_train, module_name="NeuralNet")
        logit_PB = LogitPBEstimator(
            env=env_train,
            module_name="NeuralNet",
            torso=logit_PF.module.torso,  # To share parameters between PF and PB
        )
        logZ = LogZEstimator(torch.tensor(0.0))
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        forward_actions_sampler = DiscreteActionsSampler(estimator=logit_PF, temperature=args.forward_temperature, epsilon=args.forward_epsilon)
        forward_trajectories_sampler = TrajectoriesSampler(env=env_train, actions_sampler=forward_actions_sampler)
        if args.orderinged:
            loss_fn = orderingedTrajectoryBalance(parametrization=parametrization, KL_weight=args.KL_weight, CE_weight=args.CE_weight)
        else:
            loss_fn = KLTrajectoryBalance(parametrization=parametrization, KL_weight=args.KL_weight)

    elif args.loss_type == 'FM':
        logit_F = LogEdgeFlowEstimator(env=env_train, module_name="NeuralNet")
        parametrization = FMParametrization(logit_F)
        forward_actions_sampler = DiscreteActionsSampler(estimator=logit_F, temperature=args.forward_temperature, epsilon=args.forward_epsilon)
        forward_trajectories_sampler = TrajectoriesSampler(env=env_train, actions_sampler=forward_actions_sampler)
        if args.orderinged:
            loss_fn = orderingedFlowMatching(parametrization=parametrization, alpha=args.CE_weight)
        else:
            loss_fn = FlowMatching(parametrization=parametrization)

    elif args.loss_type == 'DB':
        logit_PF = LogitPFEstimator(env=env_train, module_name="NeuralNet")
        logit_PB = LogitPBEstimator(
            env=env_train,
            module_name="NeuralNet",
            torso=logit_PF.module.torso,  # To share parameters between PF and PB
        )
        logF = LogStateFlowEstimator(
            env=env_train, 
            module_name="NeuralNet",
            torso=logit_PF.module.torso,
        )
        parametrization = DBParametrization(logit_PF, logit_PB, logF)
        forward_actions_sampler = DiscreteActionsSampler(estimator=logit_PF, temperature=args.forward_temperature, epsilon=args.forward_epsilon)
        forward_trajectories_sampler = TrajectoriesSampler(env=env_train, actions_sampler=forward_actions_sampler)
        if args.orderinged:
            loss_fn = orderingedDetailedBalance(parametrization=parametrization, KL_weight=args.KL_weight, CE_weight=args.CE_weight)
        else:
            loss_fn = KLDetailedBalance(parametrization=parametrization, KL_weight=args.KL_weight)
    
    elif args.loss_type == 'subTB':
        logit_PF = LogitPFEstimator(env=env_train, module_name="NeuralNet")
        logit_PB = LogitPBEstimator(env=env_train, module_name="NeuralNet", torso=logit_PF.module.torso)
        logF = LogStateFlowEstimator(env=env_train, module_name="NeuralNet", torso=logit_PF.module.torso, forward_looking=args.forward_looking)
        parametrization = SubTBParametrization(logit_PF, logit_PB, logF)
        forward_actions_sampler = DiscreteActionsSampler(estimator=logit_PF)
        forward_trajectories_sampler = TrajectoriesSampler(env=env_train, actions_sampler=forward_actions_sampler)
        if not args.orderinged:
            loss_fn = SubTrajectoryBalance(parametrization=parametrization) # subTB w/wo forward_looking
        else:
            if not args.forward_looking: # orderinged subTB without forward_looking
                loss_fn = orderingedSubTrajectoryBalance(parametrization=parametrization, KL_weight=args.KL_weight, CE_weight=args.CE_weight)
            else: # orderinged subTB without forward_looking, redefine something
                logF = LogStateFlowEstimator(env=env_train, module_name="NeuralNet", torso=logit_PF.module.torso)
                logR = LogStateFlowEstimator(env=env_train, module_name="NeuralNet", torso=logit_PF.module.torso)
                parametrization = FLorderingedSubTBParametrization(logit_PF, logit_PB, logF, logR)
                loss_fn = FLorderingedSubTrajectoryBalance(parametrization=parametrization, KL_weight=args.KL_weight, CE_weight=args.CE_weight)

    return parametrization, loss_fn

def get_optimizer(parametrization):
    params = [
        {
            "params": list(set(
                val for key, val in parametrization.parameters.items() if "logZ" not in key
            )),
            "lr": 0.001,
        },
        {"params": list(set(val for key, val in parametrization.parameters.items() if "logZ" in key)), "lr": 0.1},
    ]
    optimizer = torch.optim.Adam(params)
    return optimizer



def train(env_train: NAS, env_valid: NAS, args):
    if args.backward_augment:
        assert args.loss_type in ['TB', 'DB', 'subTB'], 'Only support TB, DB, and subTB.'

    for round in range(args.forward_sample_round): # update training dataset                
        train_n_steps = args.train_n_steps
        if round % 10 == 0:
            # reinitilize
            train_n_steps = args.train_n_steps_init
            parametrization, loss_fn = prepare_training(env_train, env_valid, args)
            optimizer = get_optimizer(parametrization)
            forward_actions_sampler = DiscreteActionsSampler(estimator=parametrization.logit_PF if args.loss_type !='FM' else parametrization.logF, 
                                                             temperature=args.forward_temperature, epsilon=args.forward_epsilon)
            forward_trajectories_sampler = TrajectoriesSampler(env=env_train, actions_sampler=forward_actions_sampler)
            if round == 0:
                if args.proxy:
                    proxy_sampler = ProxyTrajectoriesSampler(env_train, forward_actions_sampler, parametrization)
                    trajectories = proxy_sampler.proxy_sample(args.forward_sample_init_round, 10*args.forward_sample_init_round)
                else:
                    trajectories = forward_trajectories_sampler.sample(args.forward_sample_init_round) # init dataset size
                forward_sampled_last_states = trajectories.last_states

        objective = trajectories_to_objective(trajectories, args)
        for i in (pbar := tqdm(range(train_n_steps))):
            optimizer.zero_grad()
            loss = loss_fn(objective)
            loss.backward()
            optimizer.step()
            if i % 25 == 0:
                pbar.set_postfix({"loss": loss.item()})
        
        if args.backward_augment:
            backward_action_sampler = BackwardDiscreteActionsSampler(estimator=parametrization.logit_PB, 
                                                                     temperature=args.backward_temperature, epsilon=args.backward_epsilon)
            backward_trajectory_sampler = TrajectoriesSampler(env=env_train, actions_sampler=backward_action_sampler)
            selected_states = prioritized_replay(forward_sampled_last_states, env=env_train)
            backward_augmented_trajectories = backward_trajectory_sampler.sample_trajectories(states=selected_states, 
                                                                                              n_trajectories_per_terminal=args.backward_augment_per_terminal)
            backward_augmented_trajectories = Trajectories.revert_backward_trajectories(backward_augmented_trajectories)
            for _ in range(args.backward_augment_training_steps):
                optimizer.zero_grad()
                loss = loss_fn(trajectories_to_objective(backward_augmented_trajectories, args))
                loss.backward()
                optimizer.step()

        if args.proxy:
            new_forward_sampled_trajectories = proxy_sampler.proxy_sample(args.forward_sample_per_round, 10*args.forward_sample_per_round)
        else:
            new_forward_sampled_trajectories = forward_trajectories_sampler.sample(args.forward_sample_per_round)
        trajectories.extend(new_forward_sampled_trajectories)
        forward_sampled_last_states.extend(new_forward_sampled_trajectories.last_states)
    
    return forward_sampled_last_states, parametrization

def prioritized_replay(states: States, env: NAS, size: int = 20, alpha: float = 0.5, beta: float = 0.1):
    states_tensor = states.states_tensor
    all_num = states_tensor.shape[0]
    size = min(all_num, size)
    _, orderinged_indices = torch.sort(env.reward(states), descending=True)
    top_num_to_choose = int(all_num*beta); top_num = min(int(size*alpha), top_num_to_choose) 
    from_top = np.random.choice(range(top_num_to_choose), top_num, replace=False)
    from_bottom = np.random.choice(range(top_num_to_choose, all_num), size-top_num, replace=False)
    from_all = np.append(from_top, from_bottom)
    return states[orderinged_indices[from_all]]
    

def validation(env_valid: NAS, forward_trajectories_sampler: TrajectoriesSampler, args):
    valid_accs = []
    for _ in range(args.valid_repeat):
        max_acc = 0
        valid_accs.append(list())
        sampled_trajectories = forward_trajectories_sampler.sample(n_trajectories=args.valid_num)
        terminals = sampled_trajectories.states.states_tensor[sampled_trajectories.when_is_done-1, torch.arange(args.valid_num)]
        for terminal in terminals:
            max_acc = max(max_acc, env_valid.accuracy(terminal))
            valid_accs[-1].append(max_acc)
    valid_mu, valid_sigma = np.mean(np.array(valid_accs), axis=0), np.std(np.array(valid_accs), axis=0)
    return valid_mu, valid_sigma



def query_info(env_train: NAS, env_valid: NAS, states: States, args):
    train_accs_ind = []; valid_accs_train = []; valid_accs_valid = []; wall_clocks = []
    max_acc_train = 0.0; wall_clock = 0.0; max_acc_ind = 0; max_acc_val = 0
    states_tensor = states.states_tensor
    for i, state_tensor in enumerate(states_tensor):
    
        current_accuracy_train = env_train.accuracy(state_tensor)
        if current_accuracy_train > max_acc_train:
            max_acc_ind = i
            max_acc_train = current_accuracy_train
        train_accs_ind.append(max_acc_ind)

        current_accuracy_valid = env_valid.accuracy(state_tensor)
        max_acc_val = max(max_acc_val, current_accuracy_valid)
        valid_accs_valid.append(max_acc_val)
        
        index = env_train._api.archstr2index[env_train.state2str(state_tensor)]
        train_info = env_train._api.get_more_info(index, env_train.dataset, hp=env_train.hp, is_random=True)
        if args.dataset != 'cifar10':
            wall_clock += train_info["train-all-time"] + train_info["valid-per-time"]
        else:
            tmp_info = env_train._api.get_more_info(index, 'cifar10-valid', hp=env_train.hp, is_random=True)
            wall_clock += tmp_info["train-all-time"] + tmp_info["valid-per-time"]
        wall_clocks.append(wall_clock)
    
    for ind in train_accs_ind:
        valid_accs_train.append(env_valid.accuracy(states_tensor[ind]))
        
    all_archs = []
    for ind in train_accs_ind:
        all_archs.append(env_train._api.archstr2index[env_train.state2str(states_tensor[ind])])
    
    return all_archs, valid_accs_train, valid_accs_valid, wall_clocks


def trajectories_to_objective(trajectories: Trajectories, args):
    if args.loss_type in ['TB', 'subTB']:
        return trajectories
    elif args.loss_type == 'FM':
        return trajectories.to_non_initial_intermediary_and_terminating_states()
    elif args.loss_type == 'DB':
        return trajectories.to_transitions()
    else:
        raise NotImplementedError
    

def surrogate_log_reward(parametrization: Parametrization, state: States, args):
    # using surrogate log reward to improve sampling efficiency.
    return None



if __name__ == "__main__":
    args = get_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.CE_weight == 0:
        args.orderinged = False
    else:
        args.orderinged = True
    print("==========Printing Argument==========")
    print(args)
    ndim=args.ndim; nop=args.nop
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = OneHotNASPreprocessor(ndim=ndim, nop=nop)
    env_train = NAS(ndim=ndim, nop=nop, dataset=args.dataset, device_str=args.device, preprocessor=preprocessor, hp=args.train_hp, beta=args.beta)
    env_valid = NAS(ndim=ndim, nop=nop, dataset=args.dataset, device_str=args.device, preprocessor=preprocessor, hp=args.valid_hp, beta=1.0)

    print("==========Training and Evaluating Sampler==========")
    all_infos = {}
    for trial in range(args.forward_sample_trial):
        print(f"==========Start Trial {trial}==========")
        last_states, parametrization = train(env_train, env_valid, args)
        all_archs, valid_accs_train, valid_accs_valid, wall_clocks = query_info(env_train, env_valid, last_states, args)
        all_infos[trial] = {
            'last_states': last_states.states_tensor.detach().cpu(),
            'all_archs': all_archs,
            'all_accs_train': valid_accs_train,
            'all_accs_valid': valid_accs_valid,
            'all_total_times': wall_clocks,
        }

    logs_dir = os.path.join(args.log_dir, args.dataset, args.loss_type)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
    para_dir = logs_dir + '/' + args.info_file + '_pt'
    if not os.path.isdir(para_dir):
        os.makedirs(para_dir)
    torch.save(all_infos, os.path.join(logs_dir, args.info_file))
    parametrization.save_state_dict(para_dir)
