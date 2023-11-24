# GFN for MOO

### Code references
Our implementation is based on [gflownet](https://github.com/recursionpharma/gflownet). We add the OP-GFN implementation. 


### HyperGrid
```bash
# Runn the experiments. Visualize the learned reward.
python hypergrid_comb.py
```

### Fragment-Based Molecule Generation
We should run all experiments with at least 3 different random seeds.
```bash
declare -a OBJArray=('seh qed' 'seh sa' 'seh mw' 'qed sa' 'qed mw' 'sa mw')

for obj in ${OBJArray[@]}; do
# preference-conditioning
python -m gflownet.tasks.seh_frag_moo --objectives $obj --type pref
# goal-conditioning
python -m gflownet.tasks.seh_frag_moo --objectives $obj --type goal --replay
# order-preserving
python -m gflownet.tasks.seh_frag_moo --objectives $obj --type ordering --replay
done
```