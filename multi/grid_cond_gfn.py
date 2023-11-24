import argparse
import gzip
import itertools
import os
import pickle  # nosec B403
from collections import defaultdict
from itertools import chain, count

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from botorch.utils.multi_objective import pareto
from copy import deepcopy

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default="results/example_branincurrin.pkl.gz", type=str)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--progress", action="store_true")  # Shows a tqdm bar

# GFN
parser.add_argument("--method", default="flownet_tb", type=str)
parser.add_argument("--learning_rate", default=1e-2, help="Learning rate", type=float)
parser.add_argument("--opt", default="adam", type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--mbsize", default=128, help="Minibatch size", type=int)
parser.add_argument("--n_hid", default=64, type=int)
parser.add_argument("--n_layers", default=3, type=int)
parser.add_argument("--n_train_steps", default=5000, type=int)
parser.add_argument("--moo_strategy", default="preference", choices=["preference", "ordering-ce"])
parser.add_argument("--mask", default=None)

# Measurement
parser.add_argument("--n_distr_measurements", default=50, type=int)

# Training
parser.add_argument("--n_mp_procs", default=4, type=int)

# Env
parser.add_argument("--func", default="BraninCurrin")
parser.add_argument("--horizon", default=32, type=int)

_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])  # noqa
tl = lambda x: torch.LongTensor(x).to(_dev[0])  # noqa

class ReplayBuffer(object):
    def __init__(self, rng: np.random.Generator):
        self.capacity = 10000
        self.buffer = []
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


def currin(x):
    x_0 = x[..., 0] / 2 + 0.5
    x_1 = x[..., 1] / 2 + 0.5
    factor1 = 1 - np.exp(-1 / (2 * x_1 + 1e-10))
    numer = 2300 * x_0**3 + 1900 * x_0**2 + 2092 * x_0 + 60
    denom = 100 * x_0**3 + 500 * x_0**2 + 4 * x_0 + 20
    return factor1 * numer / denom / 13.77  # Dividing by the max to help normalize


def branin(x):
    x_0 = 15 * (x[..., 0] / 2 + 0.5) - 5
    x_1 = 15 * (x[..., 1] / 2 + 0.5)
    t1 = x_1 - 5.1 / (4 * np.pi**2) * x_0**2 + 5 / np.pi * x_0 - 6
    t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x_0)
    return 1 - (t1**2 + t2 + 10) / 308.13  # Dividing by the max to help normalize

def sphere(x):
    return (x**2).sum(axis=-1)/2

def beale(x):
    x_0 = x[..., 0]
    x_1 = x[..., 1]
    return ((1.5-x_0+x_0*x_1)**2+(2.25-x_0+x_0*x_1**2)**2+(2.625-x_0+x_0*x_1**3)**2)/38.8 # Dividing by the max to help normalize


def shubert(x):
    x_0 = (-x[..., 0])*1.5
    x_1 = (-x[..., 1])*1.5
    sum1 = sum2 = 0
    for ii in range(1,6):
        new1 = ii * np.cos((ii+1)*x_0+ii)
        new2 = ii * np.cos((ii+1)*x_1+ii)
        sum1 = sum1 + new1
        sum2 = sum2 + new2
    return (sum1 * sum2 + 186.8)/397 # Dividing by the max to help normalize


class GridEnv:
    def __init__(self, horizon, ndim=2, xrange=[-1, 1], funcs=None, obs_type="one-hot", moo_strategy="preference", mask=None):
        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.width = xrange[1] - xrange[0]
        self.funcs = [lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01] if funcs is None else funcs
        self.num_cond_dim = len(self.funcs) + 1
        self.xspace = np.linspace(*xrange, horizon)
        self._true_density = None
        self.obs_type = obs_type
        self.moo_strategy = moo_strategy
        self.mask = mask
        assert self.moo_strategy in ["preference", "ordering-ce"]
        if obs_type == "one-hot":
            self.num_obs_dim = self.horizon * self.ndim
        elif obs_type == "scalar":
            self.num_obs_dim = self.ndim
        elif obs_type == "tab":
            self.num_obs_dim = self.horizon**self.ndim

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros(self.num_obs_dim + self.num_cond_dim)
        if self.obs_type == "one-hot":
            z = np.zeros((self.horizon * self.ndim + self.num_cond_dim), dtype=np.float32)
            z[np.arange(len(s)) * self.horizon + s] = 1
        elif self.obs_type == "scalar":
            z[: self.ndim] = self.s2x(s)
        elif self.obs_type == "tab":
            idx = (s * (self.horizon ** np.arange(self.ndim))).sum()
            z[idx] = 1
        z[-self.num_cond_dim :] = self.cond_obs
        return z

    def s2x(self, s):
        return s / (self.horizon - 1) * self.width + self.start

    def s2r(self, s):
        if self.moo_strategy in ["preference"]:
            return (self.coefficients * self.s2flatr(s)).sum(-1) ** self.temperature
        elif "ordering-" in self.moo_strategy:
            return self.s2flatr(s)
    
    def s2flatr(self, s):
        x = self.s2x(s)
        flatr = np.stack([f(x) for f in self.funcs]).T
        if self.mask is None:
            return flatr
        x = flatr[...,0]; y = flatr[...,1]
        mask_r = self.mask(x,y)
        return (flatr.T*mask_r).T + 1e-8

    def reset(self, coefs=None, temp=None):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        self.coefficients = np.random.dirichlet([1.5] * len(self.funcs)) if coefs is None else coefs
        self.temperature = np.random.gamma(16, 1) if temp is None else temp
        self.cond_obs = np.concatenate([self.coefficients, [self.temperature]])
        return self.obs(), self.s2r(self._state), self._state

    def parent_transitions(self, s, used_stop_action):
        if used_stop_action:
            return [self.obs(s)], [self.ndim]
        parents = []
        actions = []
        for i in range(self.ndim):
            if s[i] > 0:
                sp = s + 0
                sp[i] -= 1
                if sp.max() == self.horizon - 1:  # can't have a terminal parent
                    continue
                parents += [self.obs(sp)]
                actions += [i]
        return parents, actions

    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.s2r(s), done, s

    def state_info(self):
        all_int_states = np.float32(list(itertools.product(*[list(range(self.horizon))] * self.ndim)))
        state_mask = (all_int_states == self.horizon - 1).sum(1) <= 1
        pos = all_int_states[state_mask].astype("float")
        x = pos / (self.horizon - 1) * (self.xspace[-1] - self.xspace[0]) + self.xspace[0]
        r = self.s2flatr(pos)
        return x, r, pos
    
    def pareto(self):
        _, r, _ = self.state_info()
        pareto = []
        re = r.T
        for i in range(re.shape[1]):
            d = ((re[:, i, None] - re) < 0).prod(0).sum()
            if d == 0:
                pareto.append(re[:, i])
        pareto = np.float32(pareto)
        return pareto

    def generate_backward(self, r, s0, reset=False):
        if reset:
            self.reset(coefs=np.zeros(2))  # this e.g. samples a new temperature
        s = np.int8(s0)
        r = max(r**self.temperature, 1e-35)  # TODO: this might hit float32 limit, handle this more gracefully?
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        used_stop_action = s.max() < self.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while s.sum() > 0 or used_stop_action:
            parents, actions = self.parent_transitions(s, used_stop_action)
            if len(parents) == 0:
                import pdb

                pdb.set_trace()
            # add the transition
            traj.append([tf(np.array(i)) for i in (parents, actions, [r], self.obs(s), [done])])
            # Then randomly choose a parent state
            if not used_stop_action:
                i = np.random.randint(0, len(parents))
                a = actions[i]
                s[a] -= 1
            else:
                a = self.ndim  # the stop action
            traj[-1].append(tf(self.obs(s)))
            traj[-1].append(tf([a]).long())
            if len(traj) == 1:
                traj[-1].append(tf(self.cond_obs))
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r = 0
        return traj


def make_mlp(ls, act=nn.LeakyReLU, tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(
        *(
            sum(
                [[nn.Linear(i, o)] + ([act()] if n < len(ls) - 2 else []) for n, (i, o) in enumerate(zip(ls, ls[1:]))],
                [],
            )
            + tail
        )
    )


class FlowNet_TBAgent:
    def __init__(self, args, envs):
        self.model = make_mlp(
            [envs[0].num_obs_dim + envs[0].num_cond_dim] + [args.n_hid] * args.n_layers + [args.ndim + 1]
        )
        self.Z = make_mlp([envs[0].num_cond_dim] + [args.n_hid // 2] * args.n_layers + [1])
        self.model.to(args.dev)
        self.n_forward_logits = args.ndim + 1
        self.envs = envs
        self.ndim = args.ndim
        self.dev = args.dev
        self.moo_strategy = args.moo_strategy

    def forward_logits(self, x):
        return self.model(x)[:, : self.n_forward_logits]

    def parameters(self):
        return chain(self.model.parameters(), self.Z.parameters())
    
    def load_parameters(self, directory=None):
        results = pickle.load(gzip.open(directory, 'rb'))
        for i, j in zip(self.parameters(), results["params"]):
            i.data = torch.tensor(j).to(self.dev)
            
    def get_default_temp(self):
        if self.moo_strategy in ["preference"]:
            temp = [None]*len(self.envs)
        elif "ordering" in self.moo_strategy:
            temp = [1.0]*len(self.envs) # ordering use temperature 1. 
        return temp

    def get_default_coefs(self):
        if self.moo_strategy in ["preference"]:
            coefs = [None]*len(self.envs)
        elif "ordering-" in self.moo_strategy:
            coefs = [np.ones(len(self.envs[0].funcs))]*len(self.envs) # ordering use temperature 1. 
        return coefs
    
    def sample(self, mbsize=None, coefs=None, temp=None):
        if mbsize is None:
            mbsize = len(self.envs)
        if coefs is None:
            coefs = self.get_default_coefs()
        if temp is None:
            temp = self.get_default_temp()
        s = tf(np.float32([env.reset(coefs=coefs[i],temp=temp[i])[0] for i, env in enumerate(self.envs)]))
        done = [False] * mbsize
        done_s = [None] * mbsize

        Z = self.Z(torch.tensor([i.cond_obs for i in self.envs]).float())[:, 0]
        self._Z = Z.detach().numpy().reshape(-1)

        while not all(done):
            cat = Categorical(logits=self.model(s))
            acts = cat.sample()
            ridx = torch.tensor((np.random.uniform(0, 1, acts.shape[0]) < 0.01).nonzero()[0])
            if len(ridx):
                racts = np.random.randint(0, cat.logits.shape[1], len(ridx))
                acts[ridx] = torch.tensor(racts)
            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            for i, d in enumerate(done):
                if not d and step[m[i]][2]:
                    assert done_s[i] is None
                    done_s[i] = step[m[i]][3]
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf(np.float32([i[0] for i in step if not i[2]]))

        return done_s

    def sample_many(self, mbsize=None, coefs=None, temp=None):
        if mbsize is None:
            mbsize = len(self.envs)
        if coefs is None:
            coefs = self.get_default_coefs()
        if temp is None:
            temp = self.get_default_temp()

        s = tf(np.float32([env.reset(coefs=coefs[i],temp=temp[i])[0] for i, env in enumerate(self.envs)]))
        done = [False] * mbsize

        Z = self.Z(torch.tensor([i.cond_obs for i in self.envs]).float())[:, 0]
        self._Z = Z.detach().numpy().reshape(-1)
        fwd_prob = [[i] for i in Z]
        bck_prob = [[] for i in range(mbsize)]
        log_flatr = [[] for i in range(mbsize)]
        temp = torch.tensor([env.temperature for env in self.envs])
        # We will progressively add log P_F(s|), subtract log P_B(|s) and R(s)
        while not all(done):
            cat = Categorical(logits=self.model(s))
            acts = cat.sample()
            ridx = torch.tensor((np.random.uniform(0, 1, acts.shape[0]) < 0.01).nonzero()[0])
            if len(ridx):
                racts = np.random.randint(0, cat.logits.shape[1], len(ridx))
                acts[ridx] = torch.tensor(racts)
            logp = cat.log_prob(acts)
            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            p_a = [
                self.envs[0].parent_transitions(sp_state, a == self.ndim)
                for a, (sp, r, done, sp_state) in zip(acts, step)
            ]
            for i, (bi, lp, (_, r, d, sp)) in enumerate(zip(np.nonzero(np.logical_not(done))[0], logp, step)):
                fwd_prob[bi].append(logp[i])
                bck_prob[bi].append(torch.tensor(np.log(1 / len(p_a[i][0]))).float())
                if d:
                    if self.moo_strategy == "preference":
                        bck_prob[bi].append(torch.tensor(np.log(r)).float())
                    elif "ordering" in self.moo_strategy:
                        log_flatr[bi].append(torch.tensor(np.log(r)).float())
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf(np.float32([i[0] for i in step if not i[2]]))
        if self.moo_strategy == "preference": 
            numerator = torch.stack([sum(i) for i in fwd_prob])
            denominator = torch.stack([sum(i) for i in bck_prob])
            log_ratio = numerator - denominator
            return log_ratio
        elif self.moo_strategy == "goal-ordering":
            pred_log_r = torch.stack([sum(i) for i in fwd_prob]) - torch.stack([sum(i) for i in bck_prob])
            log_flatr = torch.stack([i[0] for i in log_flatr])
            flatr = torch.exp(log_flatr)
            log_r = flatr.sum(dim=1).log()
            cor = torch.stack([torch.tensor(env.coefficients) for env in self.envs])
            log_cosim = torch.log(torch.nn.CosineSimilarity()(flatr, cor)+1e-8)
            return pred_log_r, log_r, log_cosim
        elif "ordering-" in self.moo_strategy:
            pred_log_r = torch.stack([sum(i) for i in fwd_prob]) - torch.stack([sum(i) for i in bck_prob])
            log_flatr = torch.stack([i[0] for i in log_flatr])
            return pred_log_r, log_flatr


    def learn_from(self, it, batch):
        if self.moo_strategy == "preference":
            if type(batch) is list:
                log_ratio = torch.stack(batch, 0)
            else:
                log_ratio = batch
            loss = log_ratio.pow(2).mean()
        elif self.moo_strategy == "ordering-ce":
            pred_log_r, log_r = batch
            loss = self.ordering_loss_ce(log_r, pred_log_r)
        else:
            raise ValueError
        return loss, self._Z[0]

    def ordering_loss(self, log_rewards, pred_log_rewards):
        assert log_rewards.shape == pred_log_rewards.shape
        if len(log_rewards.shape) == 1: 
            log_rewards = log_rewards[:,None]; pred_log_rewards = pred_log_rewards[:,None]
        shuffling_indices = torch.randperm(log_rewards.shape[0])
        log_rewards = torch.cat([log_rewards, log_rewards[shuffling_indices]],dim=1)
        pred_log_rewards = torch.cat([pred_log_rewards, pred_log_rewards[shuffling_indices]],dim=1)
        tmp = ((log_rewards[:,0] > log_rewards[:,1]).float() + (log_rewards[:,0] > log_rewards[:,1] - 1e-6).float())/2.0
        log_rewards_target = torch.cat([tmp[:,None], (1-tmp)[:,None]], dim=1)

        return torch.nn.CrossEntropyLoss()(pred_log_rewards, log_rewards_target)
    
    def ordering_loss_one(self, log_rewards, pred_log_rewards):
        zero_one = pareto.is_non_dominated(log_rewards, deduplicate=False).float()
        assert zero_one.shape == pred_log_rewards.shape
        comb_zero = torch.combinations(zero_one)
        comb_pred = torch.combinations(pred_log_rewards)
        is_equal = comb_zero[...,0] == comb_zero[...,1]
        comb_zero[is_equal] = 0.5
        num_equal = sum(is_equal); num_inequal = sum(~is_equal)
        
        equal_choice = torch.randperm(num_equal)[:64]
        inequal_choice = torch.randperm(num_inequal)[:64]
        comb_zero_equal = comb_zero[is_equal]
        comb_pred_equal = comb_pred[is_equal]
        comb_zero_inequal = comb_zero[~is_equal]
        comb_pred_inequal = comb_pred[~is_equal]
        
        loss = torch.nn.CrossEntropyLoss()(comb_pred_inequal[inequal_choice], comb_zero_inequal[inequal_choice]) + \
            torch.nn.CrossEntropyLoss()(comb_pred_equal[equal_choice], comb_zero_equal[equal_choice])
        return loss
    
    def ordering_loss_bce(self, log_rewards, pred_log_rewards):
        zero_one = pareto.is_non_dominated(log_rewards, deduplicate=False).float()
        loss = torch.nn.BCEWithLogitsLoss()(pred_log_rewards, zero_one)
        return loss
    
    def ordering_loss_ce(self, log_rewards, pred_log_rewards):
        zero_one = pareto.is_non_dominated(log_rewards, deduplicate=False).float()
        loss = torch.nn.CrossEntropyLoss()(pred_log_rewards, zero_one/sum(zero_one))
        return loss
    
    
def make_opt(params, args):
    params = list(params)
    if not len(params):
        return None
    if args.opt == "adam":
        opt = torch.optim.Adam(params, args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=1e-4)
    elif args.opt == "msgd":
        opt = torch.optim.SGD(params, args.learning_rate, momentum=args.momentum)
    return opt


def compute_exact_dag_distribution(envs, agent, args):
    env = envs[0]
    stack = [np.zeros(env.ndim, dtype=np.int32)]
    state_prob = defaultdict(lambda: np.zeros(len(envs)))
    state_prob[tuple(stack[0])] += 1
    end_prob = {}
    opened = {}
    softmax = nn.Softmax(1)
    asd = tqdm(total=env.horizon**env.ndim, disable=not args.progress or 1, leave=False)
    while len(stack):
        asd.update(1)
        s = stack.pop(0)
        p = state_prob[tuple(s)]
        if s.max() >= env.horizon - 1:
            end_prob[tuple(s)] = p
            continue
        policy = softmax(agent.forward_logits(torch.tensor(np.float32([i.obs(s) for i in envs])))).detach().numpy()
        end_prob[tuple(s)] = p * policy[:, -1]
        for i in range(env.ndim):
            sp = s + 0
            sp[i] += 1
            state_prob[tuple(sp)] += policy[:, i] * p
            if tuple(sp) not in opened:
                opened[tuple(sp)] = 1
                stack.append(sp)
    asd.close()
    all_int_states = np.int32(list(itertools.product(*[list(range(env.horizon))] * env.ndim)))
    state_mask = (all_int_states == env.horizon - 1).sum(1) <= 1
    distribution = np.float32([end_prob[i] for i in map(tuple, all_int_states[state_mask])])
    return distribution



def main(args, fs=None):
    args.dev = torch.device(args.device)
    args.ndim = 2  # Force this for Branin-Currin
    if fs is None:
        fs = [branin, currin]
    envs = [GridEnv(args.horizon, args.ndim, funcs=fs, moo_strategy=args.moo_strategy, mask=args.mask) for i in range(args.mbsize)]

    agent = FlowNet_TBAgent(args, envs)
    for i in agent.parameters():
        i.grad = torch.zeros_like(i)
    agent.model.share_memory()
    agent.Z.share_memory()


    opt = make_opt(agent.model.parameters(), args)
    optZ = make_opt(agent.Z.parameters(), args)



    all_losses = []
    distributions = []
    progress_bar = tqdm(range(args.n_train_steps + 1))
    buffer = ReplayBuffer(rng=np.random.default_rng(12345))
    for t in progress_bar:
        if len(all_losses) >= 100:
            progress_bar.set_description_str(
                " ".join([f"{np.mean([i[j] for i in all_losses[-100:]]):.5f}" for j in range(len(all_losses[0]))])
            )
        opt.zero_grad()
        optZ.zero_grad()
        data = agent.sample_many(mbsize=args.mbsize)
        losses = agent.learn_from(-1, data)  # returns (opt loss, *metrics)
        losses[0].backward()
        all_losses.append([losses[0].item()] + list(losses[1:]))
        opt.step()
        optZ.step()

    results = {
        "losses": np.float32(all_losses),
        "params": [i.data.to("cpu").numpy() for i in agent.parameters()],
        "distributions": distributions,
        "final_distribution": None,
        "cond_confs": None,
        "state_info": envs[0].state_info(),
        "args": args,
    }
    if args.save_path is not None:
        root = os.path.split(args.save_path)[0]
        if len(root):
            os.makedirs(root, exist_ok=True)
        pickle.dump(results, gzip.open(args.save_path, "wb"))
    else:
        return results


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
