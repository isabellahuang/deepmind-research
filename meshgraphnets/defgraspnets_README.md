# DefGraspNets dataset

This codebase can be used to train the DefGraspNets model in our paper [DefGraspNets: Grasp Planning on 3D Fields with Graph Neural Nets](https://arxiv.org/abs/2303.16138).

The following document describes how to train the mode, plot losses, do full predictive rollouts on test sets, and visualize those rollouts. 

## Setup
Make sure you are on the ```migrate_tf2``` branch of this fork. Then, run 
```pip install -r meshgraphnets/requirements.txt```

This was tested on Ubuntu 18.04 with Cuda 11.2. Please see this table for TensorFlow compatibility for your machine: https://www.tensorflow.org/install/source#gpu

## Dataset
Our ground truth dataset is generated using the FEM-based simulator DefGraspSim. 
This dataset contains 90 unique object. For each object, 100 random grasps are simulated. 
For each grasp, there are 50 grasp states for grasp forces between 0 and 15 N. 
Each grasp state includes the full state information of the interaction (e.g., all mesh nodal positions and element stresses).
This dataset, titled ```dgs_dataset```, can be downloaded here: https://drive.google.com/file/d/1UKrPv5eetRbXEwVMjTgtarj4T8bV4_fq/view?usp=sharing
Note that this dataset currently only contains experiments for objects with elastic modulus 5e5. More data to be uploaded. 


## Training
Test and train sets are split on the object level. By current convention, files that have a "test" in the filename in ```dgs_dataset``` are the objects in the test set (these were split randomly). 
On the ```meshgraphnets/``` level, run the following: 

```python -m meshgraphnets.run_model --dataset_dir=[path to dgs_dataset]/5e5_pd --model=deforming_plate --num_epochs_per_validation=2 --checkpoint_dir=meshgraphnets/data/chk/[my_checkpoint_dir] --num_epochs=1000 --learning_rate=1e-5 --loss_function=mse --eager=False --batch_size=1 --mode=train```

See ```meshgraphnets/run_model.py``` for more flags that can be set. 


## Plot losses
Training continually outputs train and test losses (both in normalized and real units) to a text file. You can visualize the loss plots with the following:
```python -m meshgraphnets.plot_losses --checkpoint_dir=my_checkpoint_dir2```


## Evaluate rollouts
Once training is complete, you can evaluate full rollouts on the test sets.
```python -m meshgraphnets.run_model --dataset_dir=[path to dgs_dataset]t/5e5_pd --model=deforming_plate --num_epochs_per_validation=2 --checkpoint_dir=meshgraphnets/data/chk/[my_checkpoint_dir] --num_epochs=1000 --learning_rate=1e-5 --loss_function=mse --eager=False --batch_size=1 --mode=eval --num_eval_trajectories=[number of trajectories you want to evaluate per test object]```

This process saves all trajectories into ```rollout.pkl``` in your checkpoint directory.

## Visualize rollouts
The rollouts can be visualized with the following:

```python -m meshgraphnets.plot_deforming_plate --rollout_path=meshgraphnets/data/chk/[my_checkpoint_dir]/rollout.pkl```

