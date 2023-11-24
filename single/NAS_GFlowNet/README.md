# NAS_GFlowNet

## Getting started

This work is built upon 
1. [saleml/gfn](https://github.com/saleml/gfn) is a GFlownet package.

2. [D-X-Y/NATS-Bench](https://github.com/D-X-Y/NATS-Bench) is a benchmark for NAS algorithm. You can use `pip install nats_bench` to install the library of NATS-Bench, and you should download datasets based on the `nats_bench`'s guidance.

## Usages
We support NAS on dataset: cifar10, cifar100, ImageNet16-120, and GFN methods: TB, DB, FM, subTB. The number of seeds can be adjusted by `--forward_sample_trial`. 
```bash
for dataset in cifar10 cifar100 ImageNet16-120 do
for loss in TB DB FM subTB do
# GFN-beta
python main.py --dataset $dataset --loss_type $loss_type --beta 1
# OP-GFN, with lambda_OP=0.1 (for OP-TB, lambda_OP=1 by default)
python main.py --dataset $dataset --loss_type $loss_type --CE_weight 0.1
# OP-GFN-KL, with lambda_OP=0.1, lambda_KL=0.1. 
python main.py --dataset $dataset --loss_type $loss_type --CE_weight 0.1 --KL_weight 0.1
# OP-GFN-KL-AUG, with lambda_OP=0.1, lambda_KL=0.1. 
python main.py --dataset $dataset --loss_type $loss_type --CE_weight 0.1 --KL_weight 0.1 --backward_augment
done
done
```
