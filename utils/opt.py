import os
import argparse
from pprint import pprint


class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--cuda_idx', type=str, default='cuda:0', help='cuda idx')
        self.parser.add_argument('--data_dir', type=str, default='./dataset/', help='path to dataset')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether to evaluate existing models or not')
        self.parser.add_argument('--ckpt', type=str, default='./checkpoint/', help='path to save checkpoint')               
        self.parser.add_argument('--test_id', type=int, default=1, help='id of the test participant (only used in the MoGaze dataset)')
        self.parser.add_argument('--train_sample_rate', type=int, default=2, help='sample the training data')        
        self.parser.add_argument('--save_predictions', dest='save_predictions', action='store_true', help='whether to save the prediction results or not')
        # ===============================================================
        #                     Model options
        # ===============================================================        
        self.parser.add_argument('--gcn_latent_features', type=int, default=16, help='number of latent features used in the gcn')
        self.parser.add_argument('--pose_gcn_num', type=int, default=4, help='number of gcn in the pose encoder')
        self.parser.add_argument('--fuse_gcn_num', type=int, default=16, help='number of gcn in the fuse module')
        self.parser.add_argument('--object_num_static', type=int, default=2, help='number of static objects')
        self.parser.add_argument('--object_num_dynamic', type=int, default=2, help='number of dynamic objects')
        self.parser.add_argument('--object_node_static', type=int, default=5, help='number of static nodes in the graph')
        self.parser.add_argument('--object_node_dynamic', type=int, default=5, help='number of dynamic nodes in the graph')
        self.parser.add_argument('--mlp_linear', type=int, default=128, help='size of the linear layer in the mlp')
        self.parser.add_argument('--mlp_dropout', type=float, default=0.5, help='dropout probability in the mlp')        
        self.parser.add_argument('--gcn_dropout', type=float, default=0.3, help='dropout probability in the gcn')
        self.parser.add_argument('--joint_number', type=int, default=21, help='number of joints in the human pose data')
        self.parser.add_argument('--head_node_n', type=int, default=5, help='number of head nodes in the graph')
        self.parser.add_argument('--use_dct', type=int, default=1, help='use dct or not')
        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--input_n', type=int, default=10, help='number of input frames (past frames)')
        self.parser.add_argument('--output_n', type=int, default=30, help='number of output frames (future frames)')
        self.parser.add_argument('--actions', type=str, default='all', help='actions to use')
        self.parser.add_argument('--learning_rate', type=float, default=0.01)
        self.parser.add_argument('--gamma', type=float, default=0.95, help='decay learning rate by gamma')
        self.parser.add_argument('--epoch', type=int, default=80)
        self.parser.add_argument('--velocity_loss', type=int, default=1)
        self.parser.add_argument('--test_joint_error', type=int, default=0, help='test error on each joint')
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--validation_epoch', type=int, default=10, help='interval of epoches to test')
        self.parser.add_argument('--test_batch_size', type=int, default=32)
        
    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")
        
    def parse(self, make_dir=True):
        self._initial()
        self.opt = self.parser.parse_args()        
        ckpt = self.opt.ckpt
        if make_dir==True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)                                       
        self._print()
        return self.opt