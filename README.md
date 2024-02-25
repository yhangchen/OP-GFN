# Order-Preserving GFlowNets

[Order-Preserving GFlowNets](https://arxiv.org/abs/2310.00386)

Yihang Chen, Lukas Mauch

Generative Flow Networks (GFlowNets) have been introduced as a method to sample a diverse set of candidates with probabilities proportional to a given reward. However, GFlowNets can only be used with a predefined scalar reward, which can be either computationally expensive or not directly accessible, in the case of multi-objective optimization (MOO) tasks for example. Moreover, to prioritize identifying high-reward candidates, the conventional practice is to raise the reward to a higher exponent, the optimal choice of which may vary across different environments. To address these issues, we propose Order-Preserving GFlowNets (OP-GFNs), which sample with probabilities in proportion to a learned reward function that is consistent with a provided (partial) order on the candidates, thus eliminating the need for an explicit formulation of the reward function. We theoretically prove that the training process of OP-GFNs gradually sparsifies the learned reward landscape in single-objective maximization tasks. The sparsification concentrates on candidates of a higher hierarchy in the ordering, ensuring exploration at the beginning and exploitation towards the end of the training. We demonstrate OP-GFN's state-of-the-art performance in single-objective maximization (totally ordered) and multi-objective Pareto front approximation (partially ordered) tasks, including synthetic datasets, molecule generation, and neural architecture search.


### Code Structure
- In `./single`, we provide the implementation of single-objective neural architecture search and molecular design problems.  
- In `./multi`, we provide the implementation of multi-objective hypergrid and fragment-based molecular design problems.


### Dependency
```bash
conda env create -f environment.yml
```

### Citation
```bibtex
@inproceedings{
chen2024orderpreserving,
title={Order-Preserving {GF}lowNets},
author={Yihang Chen and Lukas Mauch},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=VXDPXuq4oG}
}
```
