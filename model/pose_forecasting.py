from torch import nn
import torch
from model import graph_convolution_network
import utils.util as util


class pose_forecasting(nn.Module):
    def __init__(self, opt):
        super(pose_forecasting, self).__init__()        
        self.opt = opt        
        self.gcn_latent_features = opt.gcn_latent_features        
        self.joint_node_n = opt.joint_number       
        self.input_n = opt.input_n
        self.output_n = opt.output_n
        seq_len = self.input_n+self.output_n
        self.use_dct = opt.use_dct                        
        mlp_dropout = opt.mlp_dropout
        gcn_dropout = opt.gcn_dropout        
        pose_gcn_num = opt.pose_gcn_num
        fuse_gcn_num = opt.fuse_gcn_num
        self.object_num_dynamic = opt.object_num_dynamic
        self.object_num_static = opt.object_num_static
        self.object_node_dynamic = opt.object_node_dynamic
        self.object_node_static = opt.object_node_static
        self.head_node_n = opt.head_node_n
        mlp_linear = opt.mlp_linear
        
        self.pose_encoder = graph_convolution_network.graph_convolution_network_encoder(in_features=3, latent_features=self.gcn_latent_features, node_n=self.joint_node_n, seq_len=seq_len, p_dropout=gcn_dropout, residual_gcns_num=pose_gcn_num)
        
        self.pose_decoder = graph_convolution_network.graph_convolution_network_decoder(out_features=3, latent_features=self.gcn_latent_features, node_n=self.joint_node_n + self.object_node_dynamic + self.object_node_static + self.head_node_n, seq_len=seq_len, p_dropout=gcn_dropout, residual_gcns_num=fuse_gcn_num)
                
        self.object_point_num = 8
        self.object_fea_num = self.gcn_latent_features
        self.object_static_mlp = nn.Sequential(
            nn.Linear(3*self.object_point_num*self.object_num_static, mlp_linear),
            nn.LayerNorm([seq_len, mlp_linear], elementwise_affine=True),
            nn.Tanh(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(mlp_linear, mlp_linear),
            nn.LayerNorm([seq_len, mlp_linear], elementwise_affine=True),
            nn.Tanh(),
            nn.Dropout(p = mlp_dropout),            
            nn.Linear(mlp_linear, self.object_fea_num),
            nn.LayerNorm([seq_len, self.object_fea_num], elementwise_affine=True),
            nn.Tanh(),
            )
            
        self.object_dynamic_mlp = nn.Sequential(
            nn.Linear(3*self.object_point_num*self.object_num_dynamic, mlp_linear),
            nn.LayerNorm([seq_len, mlp_linear], elementwise_affine=True),
            nn.Tanh(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(mlp_linear, mlp_linear),
            nn.LayerNorm([seq_len, mlp_linear], elementwise_affine=True),
            nn.Tanh(),
            nn.Dropout(p = mlp_dropout),            
            nn.Linear(mlp_linear, self.object_fea_num),
            nn.LayerNorm([seq_len, self.object_fea_num], elementwise_affine=True),
            nn.Tanh(),
            )
            
        self.head_mlp = nn.Sequential(
            nn.Linear(3, mlp_linear),
            nn.LayerNorm([seq_len, mlp_linear], elementwise_affine=True),
            nn.Tanh(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(mlp_linear, mlp_linear),            
            nn.LayerNorm([seq_len, mlp_linear], elementwise_affine=True),
            nn.Tanh(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(mlp_linear, self.object_fea_num),            
            nn.LayerNorm([seq_len, self.object_fea_num], elementwise_affine=True),
            nn.Tanh(),
            )
            
        dct_m, idct_m = util.get_dct_matrix(seq_len)
        self.dct_m = torch.from_numpy(dct_m).float().to(self.opt.cuda_idx)
        self.idct_m = torch.from_numpy(idct_m).float().to(self.opt.cuda_idx)
        
    def forward(self, src, input_n=10, output_n=30):
        idx = list(range(input_n)) + [input_n -1] * output_n
        src = src[:, idx].clone()
        bs, seq_len, features = src.shape
        if self.use_dct:
            src = torch.matmul(self.dct_m, src)
        pose_input = src.clone()[:, :, :self.joint_node_n*3].permute(0, 2, 1)        
        pose_input = pose_input.reshape(bs, self.joint_node_n, 3, input_n+output_n).permute(0, 2, 1, 3)
        head_input = src.clone()[:, :, self.joint_node_n*3:self.joint_node_n*3+3]
        head_features = self.head_mlp(head_input).permute(0, 2, 1)        
        head_features = head_features.reshape(bs, self.object_fea_num, 1, input_n+output_n)
        if self.head_node_n > 0:
            head_features = head_features.expand(-1, -1, self.head_node_n, -1).clone()
        
        objects_dynamic = src.clone()[:, :, self.joint_node_n*3+3:self.joint_node_n*3+3+self.object_point_num*3*self.object_num_dynamic]                
        objects_dynamic_features = self.object_dynamic_mlp(objects_dynamic).permute(0, 2, 1)       
        objects_dynamic_features = objects_dynamic_features.reshape(bs, self.object_fea_num, 1, input_n+output_n)
        if self.object_node_dynamic > 0:
            objects_dynamic_features = objects_dynamic_features.expand(-1, -1, self.object_node_dynamic, -1).clone()
            
        objects_static = src.clone()[:, :, self.joint_node_n*3+3+self.object_point_num*3*self.object_num_dynamic:self.joint_node_n*3+3+self.object_point_num*3*self.object_num_dynamic+self.object_point_num*3*self.object_num_static]  
        objects_static_features = self.object_static_mlp(objects_static).permute(0, 2, 1)        
        objects_static_features = objects_static_features.reshape(bs, self.object_fea_num, 1, input_n+output_n)
        if self.object_node_static > 0:
            objects_static_features = objects_static_features.expand(-1, -1, self.object_node_static, -1).clone()
                        
        output = self.pose_encoder(pose_input)
        if self.head_node_n > 0:
            output = torch.cat((output, head_features), dim=2)
        if self.object_node_dynamic > 0:
            output = torch.cat((output, objects_dynamic_features), dim=2)
        if self.object_node_static > 0:
            output = torch.cat((output, objects_static_features), dim=2)
        output = self.pose_decoder(output)

        pose_output = output[:, :, :self.joint_node_n, :] + pose_input
        pose_output = pose_output.permute(0, 2, 1, 3).reshape(bs, -1, input_n+output_n).permute(0, 2, 1)
        if self.use_dct:
            pose_output = torch.matmul(self.idct_m, pose_output)
        pose_output = pose_output[:, -output_n:, :]
        
        return pose_output