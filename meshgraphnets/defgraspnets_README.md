# DefGraspNets dataset

This codebase can be used to train the DefGraspNets model in our paper [DefGraspNets: Grasp Planning on 3D Fields with Graph Neural Nets](https://arxiv.org/abs/2303.16138).

Here's an example of how you can run the training: 

```
python -m meshgraphnets.run_model --dataset_dir=../../[name_of_dataset_dir] --model=deforming_plate --num_epochs_per_validation=2 --checkpoint_dir=meshgraphnets/data/[name_of_checkpoint_dir] --gripper_force_action_input=True --noise_scale=3e-6 --noise_on_adjacent_step=True --compute_world_edges=False --predict_stress_t_only --predict_pos_change_from_initial_only --simplified_predict_stress_change_only --latent_size=128 --use_pd_stress=False --num_epochs=1000 --learning_rate=1e-5 --loss_function=mse --eager=False --batch_size=1 --mode=train
```

