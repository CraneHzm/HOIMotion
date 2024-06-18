python main_adt_pose_forecasting.py --data_dir /scratch/hu/pose_forecast/adt_hoimotion/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:3 --train_sample_rate 2 --epoch 80 --velocity_loss 1 --object_node_static 0 --object_node_dynamic 5 --head_node_n 5 --use_dct 1;

python main_adt_pose_forecasting.py --data_dir /scratch/hu/pose_forecast/adt_hoimotion/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:3 --train_sample_rate 2 --epoch 80 --velocity_loss 1 --object_node_static 0 --object_node_dynamic 5 --head_node_n 5 --use_dct 1 --is_eval --actions 'all';

python main_adt_pose_forecasting.py --data_dir /scratch/hu/pose_forecast/adt_hoimotion/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:3 --train_sample_rate 2 --epoch 80 --velocity_loss 1 --object_node_static 0 --object_node_dynamic 5 --head_node_n 5 --use_dct 1 --is_eval --actions 'work';

python main_adt_pose_forecasting.py --data_dir /scratch/hu/pose_forecast/adt_hoimotion/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:3 --train_sample_rate 2 --epoch 80 --velocity_loss 1 --object_node_static 0 --object_node_dynamic 5 --head_node_n 5 --use_dct 1 --is_eval --actions 'decoration';

python main_adt_pose_forecasting.py --data_dir /scratch/hu/pose_forecast/adt_hoimotion/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:3 --train_sample_rate 2 --epoch 80 --velocity_loss 1 --object_node_static 0 --object_node_dynamic 5 --head_node_n 5 --use_dct 1 --is_eval --actions 'meal';