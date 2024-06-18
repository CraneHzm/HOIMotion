# Extract data from the MoGaze dataset

import h5py
import numpy as np
from utils import quaternion_matrix, remake_dir, euler2xyz, euler2xyz_head
from scipy.spatial.transform import Rotation as R


# the original mogaze dataset downloaded from https://humans-to-robots-motion.github.io/mogaze/
dataset_path = "/datasets/public/zhiming_datasets/mogaze/"
dataset_processed_path = "/scratch/hu/pose_forecast/mogaze_hoimotion/"
# ["7_2"] is not used due to the low quality of eye gaze data
data_idx = ["1_1", "1_2", "2_1", "4_1", "5_1", "6_1", "6_2", "7_1", "7_3"]
original_fps = 120.0
# downsample the original data to 30.0 fps
downsample_rate = 4
# check the quality of eye gaze data
check_eye_gaze = True
# gaze confidence level >= 0.6 is considered as high-quality
confidence_level_threshold = 0.6
# a recording is used only if it contains more than 80% of high-quality eye gaze data
confidence_ratio_threshold = 0.8
# drop the beginning: each action starts with a waiting time, thus we drop sometime in the beginning
drop_beginning = False
# for each recording, we use at most the last three seconds to avoid the waiting time (no action) at the beginning
time_use = 3 # the timespan of an action is usually between two and three seconds (see the MoGaze paper)
static_objects_extracted_num = 2
dyanmic_objects_extracted_num = 2


for data in data_idx:
    print('processing data p{}'.format(data))
    data_processed_path = dataset_processed_path + "/p" + data + "/"
    remake_dir(data_processed_path)
    
    # load human pose data
    f_pose = h5py.File(dataset_path + 'p' + data + "_human_data.hdf5", 'r')
    pose = f_pose['data'][:]
    print("Human pose shape: {}".format(pose.shape))
    
    # load objects    
    f_object = h5py.File(dataset_path + 'p' + data + "_object_data.hdf5", 'r')
    objects = {}
    object_names = ['table', 'cup_red', 'shelf_laiva', 'shelf_vesken', 'plate_blue', 'jug', 'goggles', 'plate_green', 'plate_red', 'cup_green', 'cup_blue', 'red_chair', 'cup_pink', 'plate_pink', 'bowl', 'blue_chair']
    
    object_idx = 0
    for key in f_object['bodies'].keys():
        # Object Data: ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "rot_w"]
        objects[object_names[object_idx]] = f_object['bodies/' + key][:]
        object_idx = object_idx + 1        
    print("Object shape : {}".format(objects['table'].shape))
    # goggles object is used to calibrate the eye gaze data
    goggles_object = objects["goggles"]
    
    # load eye gaze
    gaze = np.zeros((pose.shape[0], 3))
    gaze_confidence = np.zeros((pose.shape[0], 1))
    with h5py.File(dataset_path + 'p' + data + "_gaze_data.hdf5", 'r') as f_gaze:
        gaze_data = f_gaze['gaze'][:, 2:5]
        confidence = f_gaze['gaze'][:, -1]
        calib = f_gaze['gaze'].attrs['calibration']
        for i in range(gaze.shape[0]):
                gaze_confidence[i] = confidence[i]                
                rotmat = quaternion_matrix(calib)
                rotmat = np.dot(quaternion_matrix(goggles_object[i, 3:7]), rotmat)
                endpos = gaze_data[i]
                # Borrowed from https://github.com/PhilippJKratzer/humoro/blob/master/humoro/player_pybullet.py
                if endpos[2] < 0:
                    endpos *= -1  # mirror gaze point if wrong direction
                endpos = np.dot(rotmat, endpos) # gaze calibration
                direction = endpos - goggles_object[i][0:3]
                direction = [x / np.linalg.norm(direction) for x in direction]
                gaze[i] = direction                       
            
    print("Eye gaze shape : {}".format(gaze.shape))
    
    # segment the data
    f_seg = h5py.File(dataset_path + 'p' + data + "_segmentations.hdf5", 'r')
    segments = f_seg['segments'][:]
    pick_num = 0
    place_num = 0
    for i in range(len(segments) - 1):
        if data == "6_1" and i == 248:  # error in the dataset, null twice in a row
            i += 1
        if data == "7_2" and i == 0:  # error in the dataset, null twice in a row
            i += 1
        
        current_segment = segments[i]
        next_segment = segments[i + 1]
        current_goal = next_segment[2].decode("utf-8")
        current_object = current_segment[2].decode("utf-8")
        
        action_time = (current_segment[1] - current_segment[0] + 1)/original_fps # seconds                
        #print("Action: {}, time: {:.1f} s".format(action, action_time))
                        
        start_frame = current_segment[0]
        end_frame = current_segment[1]
        #print("start: {}, end: {}".format(start_frame, end_frame))    
                        
        if check_eye_gaze:
            # reset the start and end frame
            for frame in range(start_frame, end_frame+1):
                if gaze_confidence[frame] >= confidence_level_threshold:
                    start_frame = frame
                    break
            for frame in range(end_frame, start_frame, -1):
                if gaze_confidence[frame] >= confidence_level_threshold:
                    end_frame = frame
                    break
                    
            # ratio of the high-quality eye gaze data    
            ratio = np.sum(gaze_confidence[start_frame:end_frame+1]>=confidence_level_threshold)/(end_frame-start_frame+1)
            if ratio >= confidence_ratio_threshold:
                # replace low-quality eye gaze data with previous high-quality one
                for frame in range(start_frame+1, end_frame):
                    if gaze_confidence[frame] < confidence_level_threshold:
                        gaze[frame] = gaze[frame-1]                
            else:
                continue
                
        if current_goal == "null":
            action = "place"
            place_num = place_num + 1
        else:
            action = "pick"
            pick_num = pick_num + 1
                                        
        # segment data
        length = end_frame-start_frame+1
        length_use = int(time_use*original_fps)
        if drop_beginning and length > length_use:
            start_frame = end_frame - length_use
            pose_seg = pose[start_frame:end_frame+1:downsample_rate, :]
            #print("Human pose segment shape: {}".format(pose_seg.shape))
            # calculate the xyz positions of human joints
            pose_xyz = euler2xyz(pose_seg)
            # calculate the head direction from human pose
            head_direction = euler2xyz_head(pose_seg)
            gaze_seg = gaze[start_frame:end_frame+1:downsample_rate, :]
            #print("Eye gaze segment shape: {}".format(gaze_seg.shape))
            objects_seg = {}
            for key in objects.keys():
                objects_seg[key] = objects[key][start_frame:end_frame+1:downsample_rate, :]            
            #print("Object segment shape : {}".format(objects_seg['cup_red'].shape))
        else:
            pose_seg = pose[start_frame:end_frame+1:downsample_rate, :]
            #print("Human pose segment shape: {}".format(pose_seg.shape))
            # calculate the xyz positions of human joints
            pose_xyz = euler2xyz(pose_seg)
            # calculate the head direction from human pose
            head_direction = euler2xyz_head(pose_seg)
            gaze_seg = gaze[start_frame:end_frame+1:downsample_rate, :]
            #print("Eye gaze segment shape: {}".format(gaze_seg.shape))
            objects_seg = {}
            for key in objects.keys():
                objects_seg[key] = objects[key][start_frame:end_frame+1:downsample_rate, :]            
            #print("Object segment shape : {}".format(objects_seg['cup_red'].shape))

        if action == "pick":
            num = pick_num
        if action == "place":
            num = place_num
            
        # objects that are dynamic/pickable
        dynamic_objects_names = ['cup_red', 'plate_blue', 'jug', 'plate_green', 'plate_red', 'cup_green', 'cup_blue','cup_pink', 'plate_pink', 'bowl']
        dynamic_objects_types = ['cup', 'plate', 'jug', 'plate', 'plate', 'cup', 'cup','cup', 'plate', 'bowl']
        dynamic_objects_range = {
        'cup': [-0.041, 0.041, -0.035, 0.035, -0.040, 0.040],
        'plate': [-0.092, 0.092, -0.092, 0.092, 0.0, 0.007],
        'jug': [-0.059, 0.065, -0.076, 0.076, -0.15, 0.14],
        'bowl': [-0.142, 0.142, -0.066, 0.040, -0.19, 0.19]
        }
        dynamic_objects_bbx_pos = {}
        for key in dynamic_objects_range:
            x_min = dynamic_objects_range[key][0]
            x_max = dynamic_objects_range[key][1]
            y_min = dynamic_objects_range[key][2]
            y_max = dynamic_objects_range[key][3]
            z_min = dynamic_objects_range[key][4]
            z_max = dynamic_objects_range[key][5]
            dynamic_objects_bbx_pos[key] = []
            dynamic_objects_bbx_pos[key].append([x_min, y_min, z_min])
            dynamic_objects_bbx_pos[key].append([x_max, y_min, z_min])
            dynamic_objects_bbx_pos[key].append([x_max, y_min, z_max])
            dynamic_objects_bbx_pos[key].append([x_min, y_min, z_max])
            dynamic_objects_bbx_pos[key].append([x_min, y_max, z_max])
            dynamic_objects_bbx_pos[key].append([x_max, y_max, z_max])
            dynamic_objects_bbx_pos[key].append([x_max, y_max, z_min])
            dynamic_objects_bbx_pos[key].append([x_min, y_max, z_min])
        
        dyanmic_objects_num = len(dynamic_objects_names)                
        dynamic_objects = np.zeros((pose_seg.shape[0], dyanmic_objects_num*3))
        for p in range(dyanmic_objects_num):
            dynamic_objects[:, p*3:p*3+3] = objects_seg[dynamic_objects_names[p]][:, 0:3]

        dynamic_objects_bbx = np.zeros((pose_seg.shape[0], dyanmic_objects_num*3*8))
        for n in range(pose_seg.shape[0]):
            for p in range(dyanmic_objects_num):
                object_name = dynamic_objects_names[p]
                pos = objects_seg[object_name][n, 0:3]
                ori = objects_seg[object_name][n, 3:7]
                rotmat = quaternion_matrix(ori)
                object_type = dynamic_objects_types[p]
                object_bbx = dynamic_objects_bbx_pos[object_type]                
                for v in range(8):
                    point = object_bbx[v]
                    point = pos + np.dot(rotmat, point)                
                    dynamic_objects_bbx[n, p*8*3+v*3:p*8*3+v*3+3] = point
                               
        dynamic_objects_extracted = np.zeros((dynamic_objects.shape[0], dyanmic_objects_extracted_num*3))
        dynamic_objects_bbx_extracted = np.zeros((dynamic_objects.shape[0], dyanmic_objects_extracted_num*3*8))
        object_importance = np.zeros((dynamic_objects.shape[0], dyanmic_objects_num))
        for i in range(dynamic_objects.shape[0]):
            head_pos = objects_seg['goggles'][i, 0:3]                       
            head_dir = head_direction[i]
            for j in range(dyanmic_objects_num):
                object_pos = dynamic_objects[i, j*3:j*3+3]
                object_direction = object_pos - head_pos
                object_direction = [x / np.linalg.norm(object_direction) for x in object_direction]               
                object_importance[i, j] = np.sum(head_dir*object_direction)
        
        for i in range(dynamic_objects.shape[0]):
            importance = object_importance[i, :]
            index = np.flip(np.argsort(importance))
            for j in range(dyanmic_objects_extracted_num):
                object_pos = dynamic_objects[i, index[j]*3:index[j]*3+3]
                object_pos_bbx = dynamic_objects_bbx[i, index[j]*3*8:index[j]*3*8+3*8]
                dynamic_objects_extracted[i, j*3:j*3+3] = object_pos
                dynamic_objects_bbx_extracted[i, j*3*8:j*3*8+3*8] = object_pos_bbx
                
        # objects that are static/not pickable
        static_objects_names = ['table', 'shelf_laiva', 'shelf_vesken', 'red_chair', 'blue_chair']
        static_objects_types = ['table', 'shelf_laiva', 'shelf_vesken', 'chair', 'chair']
        static_objects_range = {
        'table': [-0.4, 0.4, -0.4, 0.4, -0.7, 0.0],
        'shelf_laiva': [-0.31, 0.31, -0.13, 0.12, -0.8, 0.8],
        'shelf_vesken': [-0.19, 0.19, -0.12, 0.12, -0.5, 0.5],
        'chair': [-0.45, 0.09, -0.37, 0.38, -0.2, 0.2]
        }        
        static_objects_bbx_pos = {}
        for key in static_objects_range:
            x_min = static_objects_range[key][0]
            x_max = static_objects_range[key][1]
            y_min = static_objects_range[key][2]
            y_max = static_objects_range[key][3]
            z_min = static_objects_range[key][4]
            z_max = static_objects_range[key][5]
            static_objects_bbx_pos[key] = []
            static_objects_bbx_pos[key].append([x_min, y_min, z_min])
            static_objects_bbx_pos[key].append([x_max, y_min, z_min])
            static_objects_bbx_pos[key].append([x_max, y_min, z_max])
            static_objects_bbx_pos[key].append([x_min, y_min, z_max])
            static_objects_bbx_pos[key].append([x_min, y_max, z_max])
            static_objects_bbx_pos[key].append([x_max, y_max, z_max])
            static_objects_bbx_pos[key].append([x_max, y_max, z_min])
            static_objects_bbx_pos[key].append([x_min, y_max, z_min])            
        
        static_objects_num = len(static_objects_names)
        static_objects = np.zeros((pose_seg.shape[0], static_objects_num*3))
        for p in range(static_objects_num):
            static_objects[:, p*3:p*3+3] = objects_seg[static_objects_names[p]][:, 0:3]

        static_objects_bbx = np.zeros((pose_seg.shape[0], static_objects_num*3*8))
        for n in range(pose_seg.shape[0]):
            for p in range(static_objects_num):
                object_name = static_objects_names[p]
                pos = objects_seg[object_name][n, 0:3]
                ori = objects_seg[object_name][n, 3:7]
                rotmat = quaternion_matrix(ori)
                object_type = static_objects_types[p]
                object_bbx = static_objects_bbx_pos[object_type]                
                for v in range(8):
                    point = object_bbx[v]
                    point = pos + np.dot(rotmat, point)                
                    static_objects_bbx[n, p*8*3+v*3:p*8*3+v*3+3] = point
                                                       
        
        static_objects_extracted = np.zeros((static_objects.shape[0], static_objects_extracted_num*3))
        static_objects_bbx_extracted = np.zeros((static_objects.shape[0], static_objects_extracted_num*3*8))
        object_importance = np.zeros((static_objects.shape[0], static_objects_num))
        for i in range(static_objects.shape[0]):
            head_pos = objects_seg['goggles'][i, 0:3]            
            head_dir = head_direction[i]
            for j in range(static_objects_num):
                object_pos = static_objects[i, j*3:j*3+3]
                object_direction = object_pos - head_pos
                object_direction = [x / np.linalg.norm(object_direction) for x in object_direction]                                
                object_importance[i, j] = np.sum(head_dir*object_direction)
        
        for i in range(static_objects.shape[0]):
            importance = object_importance[i, :]
            index = np.flip(np.argsort(importance))
            for j in range(static_objects_extracted_num):
                object_pos = static_objects[i, index[j]*3:index[j]*3+3]
                object_pos_bbx = static_objects_bbx[i, index[j]*3*8:index[j]*3*8+3*8]
                static_objects_extracted[i, j*3:j*3+3] = object_pos
                static_objects_bbx_extracted[i, j*3*8:j*3*8+3*8] = object_pos_bbx
        
        pose_euler_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "pose_euler.npy"
        pose_xyz_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "pose_xyz.npy"
        head_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "head.npy"
        gaze_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "gaze.npy"
        objects_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "objects_all.npy"
                
        dynamic_objects_extracted_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "objects_dynamic.npy"
        dynamic_objects_bbx_extracted_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "objects_dynamicbbx.npy"

        static_objects_extracted_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "objects_static.npy"
        static_objects_bbx_extracted_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "objects_staticbbx.npy"
                        
        np.save(pose_euler_file, pose_seg)
        np.save(pose_xyz_file, pose_xyz)            
        np.save(head_file, head_direction)
        np.save(gaze_file, gaze_seg)
        np.save(objects_file, objects_seg)
        
        np.save(dynamic_objects_extracted_file, dynamic_objects_extracted)
        np.save(dynamic_objects_bbx_extracted_file, dynamic_objects_bbx_extracted)
        np.save(static_objects_extracted_file, static_objects_extracted)
        np.save(static_objects_bbx_extracted_file, static_objects_bbx_extracted)