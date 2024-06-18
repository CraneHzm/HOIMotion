from torch.utils.data import Dataset
import numpy as np
import os


class adt_dataset(Dataset):

    def __init__(self, data_dir, input_n, output_n, actions = 'all', train_flag = 1, sample_rate=1):
        actions = self.define_actions(actions)
        self.sample_rate = sample_rate
        if train_flag == 1:
            data_dir = data_dir + 'train/'
        if train_flag == 0:
            data_dir = data_dir + 'test/'
            
        self.pose_head_objects = self.load_data(data_dir, input_n, output_n, actions)
        
    def define_actions(self, action):
        """
        Define the list of actions we are using.

        Args
        action: String with the passed action. Could be "all"
        Returns
        actions: List of strings of actions
        Raises
        ValueError if the action is not included.
        """
        
        actions = ['work', 'decoration', 'meal']
        if action in actions:
            return [action]

        if action == "all":
            return actions
        raise( ValueError, "Unrecognised action: %d" % action )
                
    def load_data(self, data_dir, input_n, output_n, actions):
        action_number = len(actions)
        seq_len = input_n + output_n
        pose_head_objects = []        
        file_names = sorted(os.listdir(data_dir))
        pose_xyz_file_names = {}
        head_file_names = {}
        dynamic_objects_file_names = {}
        static_objects_file_names = {}
        for action_idx in np.arange(action_number):
            pose_xyz_file_names[actions[ action_idx ]] = []
            head_file_names[actions[ action_idx ]] = []
            dynamic_objects_file_names[actions[ action_idx ]] = []
            static_objects_file_names[actions[ action_idx ]] = []
        
        for name in file_names:
            name_split = name.split('_')
            action = name_split[2]
            if action in actions:
                data_type = name_split[-1][:-4]
                if(data_type == 'xyz'):
                    pose_xyz_file_names[action].append(name)
                if(data_type == 'head'):
                    head_file_names[action].append(name)
                if(data_type == 'dynamicbbx'):
                    dynamic_objects_file_names[action].append(name)
                if(data_type == 'staticbbx'):
                    static_objects_file_names[action].append(name)
                                
        for action_idx in np.arange(action_number):
            action = actions[ action_idx ]
            segments_number = len(pose_xyz_file_names[action])
            print("Reading action {}, segments number {}".format(action, segments_number))
                    
            for i in range(segments_number):
                pose_xyz_data_path = data_dir + pose_xyz_file_names[action][i]
                pose_xyz_data = np.load(pose_xyz_data_path)
                                                              
                num_frames = pose_xyz_data.shape[0]
                if num_frames < seq_len:
                    continue
                    
                head_data_path = data_dir + head_file_names[action][i]
                head_data = np.load(head_data_path)

                dynamic_objects_data_path = data_dir + dynamic_objects_file_names[action][i]
                dynamic_num = 2
                dynamic_objects_data = np.load(dynamic_objects_data_path)[:, :dynamic_num, :, :]
                dynamic_objects_data = dynamic_objects_data.reshape(dynamic_objects_data.shape[0], -1)

                static_objects_data_path = data_dir + static_objects_file_names[action][i]
                static_num = 2
                static_objects_data = np.load(static_objects_data_path)[:, :static_num, :, :]
                static_objects_data = static_objects_data.reshape(static_objects_data.shape[0], -1)
                
                pose_head_objects_data = pose_xyz_data
                pose_head_objects_data = np.concatenate((pose_head_objects_data, head_data), axis=1)
                pose_head_objects_data = np.concatenate((pose_head_objects_data, dynamic_objects_data), axis=1)
                pose_head_objects_data = np.concatenate((pose_head_objects_data, static_objects_data), axis=1)
                                        
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                #print(fs_sel)
                seq_sel = pose_head_objects_data[fs_sel, :]
                seq_sel = seq_sel[0::self.sample_rate, :, :]
                #print(seq_sel.shape)
                if len(pose_head_objects) == 0:
                    pose_head_objects = seq_sel
                else:
                    pose_head_objects = np.concatenate((pose_head_objects, seq_sel), axis=0)
                    
        return pose_head_objects
        
  
    def __len__(self):
        return np.shape(self.pose_head_objects)[0]

    def __getitem__(self, item):
        return self.pose_head_objects[item]

        
if __name__ == "__main__":
    data_dir = "/scratch/hu/pose_forecast/adt_hoimotion/"
    input_n = 10
    output_n = 30
    actions = 'all'
    train_sample_rate = 1
    train_flag = 0
    train_dataset = adt_dataset(data_dir, input_n, output_n, actions, train_flag, train_sample_rate)
    print("Training data size: {}".format(train_dataset.pose_head_objects.shape))