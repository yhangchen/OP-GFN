import grid_cond_gfn as gfn
import torch
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from gflownet.utils import metrics
import os, pickle, gzip, itertools
from botorch.utils.multi_objective.pareto import is_non_dominated

def hyper_2d(moo_strategy, run=True):
    all_fs = [gfn.branin, gfn.currin, gfn.shubert, gfn.beale]
    for ind_1, ind_2 in itertools.combinations(range(len(all_fs)), 2):
        if ind_1 < ind_2:
            fs = [all_fs[ind_1], all_fs[ind_2]]
            hps = gfn.parser.parse_args([])
            hps.n_train_steps = 1000 # The more steps the better
            hps.dev = torch.device(hps.device)
            hps.ndim = 2  # Force this for Branin-Currin
            hps.moo_strategy = moo_strategy
            save_path = f"{fs[0].__name__}-{fs[1].__name__}.pkl.gz"
            hps.save_path = os.path.join(f"results/results_2_{moo_strategy}", save_path)
            hps.mask = None
            if run:
                gfn.main(hps, fs)
            envs = [gfn.GridEnv(hps.horizon, hps.ndim, funcs=fs, moo_strategy=hps.moo_strategy,mask=hps.mask) for i in range(hps.mbsize)]
            pareto = envs[0].pareto()
            s,r,_ = envs[0].state_info()

            agent = gfn.FlowNet_TBAgent(hps, envs)
            agent.load_parameters(hps.save_path)
            outs = []
            for _ in range(10):
                outs.extend(agent.sample(coefs=None,temp=None))
            rewards = np.array([envs[0].s2flatr(out) for out in outs])
            idcs = is_non_dominated(torch.tensor(rewards), deduplicate=False)
            pareto_front = rewards[idcs]
            
            log_path = f"logs/logs_2_{moo_strategy}"
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            
            torch.save({
                "all_true": r,
                "pareto_true": pareto,
                "all_generated": rewards,
                "pareto_generated": pareto_front,
            }, os.path.join(log_path, save_path))

def hyper_3d(moo_strategy, run=True):
    all_fs = [gfn.branin, gfn.currin, gfn.shubert, gfn.beale]
    for ind_1, ind_2, ind_3 in itertools.combinations(range(len(all_fs)), 3):
        if ind_1 < ind_2 < ind_3:
            fs = [all_fs[ind_1], all_fs[ind_2], all_fs[ind_3]]
            hps = gfn.parser.parse_args([])
            hps.n_train_steps = 1000 # The more steps the better
            hps.dev = torch.device(hps.device)
            hps.ndim = 2  # Force this for Branin-Currin
            hps.moo_strategy = moo_strategy
            save_path = f"{fs[0].__name__}-{fs[1].__name__}-{fs[2].__name__}.pkl.gz"
            hps.save_path = os.path.join(f"results/results_3_{moo_strategy}", save_path)
            hps.mask = None
            if run:
                gfn.main(hps, fs)
            envs = [gfn.GridEnv(hps.horizon, hps.ndim, funcs=fs, moo_strategy=hps.moo_strategy,mask=hps.mask) for i in range(hps.mbsize)]
            pareto = envs[0].pareto()
            s,r,_ = envs[0].state_info()

            agent = gfn.FlowNet_TBAgent(hps, envs)
            agent.load_parameters(hps.save_path)
            outs = []
            for _ in range(10):
                outs.extend(agent.sample(coefs=None,temp=None))
            rewards = np.array([envs[0].s2flatr(out) for out in outs])
            idcs = is_non_dominated(torch.tensor(rewards), deduplicate=False)
            pareto_front = rewards[idcs]
            
            log_path = f"logs/logs_3_{moo_strategy}"
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            
            torch.save({
                "all_true": r,
                "pareto_true": pareto,
                "all_generated": rewards,
                "pareto_generated": pareto_front,
            }, os.path.join(log_path, save_path))


def hyper_4d(moo_strategy, run=True):
    all_fs = [gfn.branin, gfn.currin, gfn.shubert, gfn.beale]
    for ind_1, ind_2, ind_3, ind_4 in itertools.combinations(range(len(all_fs)), 4):
        if ind_1 < ind_2 < ind_3 < ind_4:
            fs = [all_fs[ind_1], all_fs[ind_2], all_fs[ind_3], all_fs[ind_4]]
            hps = gfn.parser.parse_args([])
            hps.n_train_steps = 1000 # The more steps the better
            hps.dev = torch.device(hps.device)
            hps.ndim = 2  # Force this for Branin-Currin
            hps.moo_strategy = moo_strategy
            save_path = f"{fs[0].__name__}-{fs[1].__name__}-{fs[2].__name__}-{fs[3].__name__}.pkl.gz"
            hps.save_path = os.path.join(f"results/results_4_{moo_strategy}", save_path)
            hps.mask = None
            if run:
                gfn.main(hps, fs)
            envs = [gfn.GridEnv(hps.horizon, hps.ndim, funcs=fs, moo_strategy=hps.moo_strategy,mask=hps.mask) for i in range(hps.mbsize)]
            pareto = envs[0].pareto()
            s,r,_ = envs[0].state_info()

            agent = gfn.FlowNet_TBAgent(hps, envs)
            agent.load_parameters(hps.save_path)
            outs = []
            for _ in range(50):
                outs.extend(agent.sample(coefs=None,temp=[16.0]*len(envs)))
            rewards = np.array([envs[0].s2flatr(out) for out in outs])
            idcs = is_non_dominated(torch.tensor(rewards), deduplicate=False)

            pareto_front = rewards[idcs]
            
            log_path = f"logs/logs_4_{moo_strategy}"
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            
            torch.save({
                "all_true": r,
                "pareto_true": pareto,
                "all_generated": rewards,
                "pareto_generated": pareto_front,
            }, os.path.join(log_path, save_path))

            
if __name__ == "__main__":
    moo_strategies = ['ordering-ce','preference']
    for m in moo_strategies:
        hyper_2d(m, run=True)
        hyper_3d(m, run=True)
        hyper_4d(m, run=True)


    count = 0
    fig, axes = plt.subplots(3,6, figsize=(12,6),layout='compressed')
    for i, axs in enumerate(axes):
        for j, ax in enumerate(axs):
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    axes[0][0].set_ylabel("True")
    axes[1][0].set_ylabel("OP-GFN")
    axes[2][0].set_ylabel("PC-GFN")


        

    all_fs = [gfn.branin, gfn.currin, gfn.shubert, gfn.beale]

    def generate_distribution(moo_strategy, fs):
        hps = gfn.parser.parse_args([])
        hps.n_train_steps = 1000 # The more steps the better
        hps.dev = torch.device(hps.device)
        hps.ndim = 2  # Force this for Branin-Currin
        hps.moo_strategy = moo_strategy
        save_path = f"{fs[0].__name__}-{fs[1].__name__}.pkl.gz"
        hps.save_path = os.path.join(f"results/results_2_{moo_strategy}", save_path)
        hps.mask = None
        
        envs = [gfn.GridEnv(hps.horizon, hps.ndim, funcs=fs, moo_strategy=hps.moo_strategy,mask=hps.mask) for i in range(hps.mbsize)]
        agent = gfn.FlowNet_TBAgent(hps, envs)
        agent.load_parameters(hps.save_path)
        
        outs = []
        for _ in range(1):
            outs.extend(agent.sample(coefs=None,temp=None))
        
        s,r,_ = envs[0].state_info()
        zero_one = is_non_dominated(torch.tensor(r), deduplicate=False).numpy()
        zero_one = np.concatenate([zero_one, [0]]).reshape((hps.horizon,hps.horizon))
        
        distributions = gfn.compute_exact_dag_distribution(envs, agent, hps)
        distributions = distributions.sum(axis=-1)/distributions.shape[-1]
        distributions = np.concatenate([distributions, [0]]).reshape((hps.horizon,hps.horizon))
        return save_path, zero_one, distributions

    for ind_1, ind_2 in itertools.combinations(range(len(all_fs)), 2):
        if ind_1 < ind_2:
            fs = [all_fs[ind_1], all_fs[ind_2]]
            save_path, zero_one, distributions = generate_distribution(moo_strategy='ordering-ce', fs=fs)
            _, _, distributions_pref = generate_distribution(moo_strategy='preference', fs=fs)
            axes[0][count].set_title(save_path[5:-7])
            im1 = axes[0][count].imshow(zero_one, cmap='Blues')
            im2 = axes[1][count].imshow(distributions, cmap='Blues')
            im3 = axes[2][count].imshow(distributions_pref, cmap='Blues')

            count += 1
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    plt.savefig('figs/dag_dis.pdf',bbox_inches='tight')
