# Molecule_GFlowNet

### Code references
Our implementation is based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet), and "Local Search GFlowNets" (https://github.com/dbsxodud-11/ls_gfn)

### Our contribution (in terms of codes)

We implement our method of OP-TB. 

### Large files

To run `sehstr` task, you should download `sehstr_gbtr_allpreds.pkl.gz` and `block_18_stop6.pkl.gz`. Both are available for download at [DOI: 10.6084/m9.figshare.22806671](https://figshare.com/articles/dataset/sEH_dataset_for_GFlowNet_/22806671). These files should be placed in `datasets/sehstr/`. 


### Main Experiments

We should run all experiments with at least 3 different random seeds.

We set the parameters for each dataset in the corresponding folder in `./exp`. 
```bash
for dataset in bag qm9str sehstr tfbind8 tfbind10 do
# Baselines
for model in a2c ppo sql mars tb db subtb maxent sub do
python runexpwb.py --setting $dataset --model $model --guide uniform  --wandb_project molecule_all --wandb_mode online
done
# OP-TB
python runexpwb.py --setting $dataset --model tb --ordering True --guide uniform  --wandb_project molecule_all --wandb_mode online
done
```