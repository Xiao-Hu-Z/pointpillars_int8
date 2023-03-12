import sys
sys.path.append("..")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime
import numpy as np
import argparse

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm

        if self.use_norm: # True
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        return x_max

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
             num_point_features += 1
        self.num_point_features = num_point_features
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(self, features, **kwargs):
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()

        return features

def build_vfe(ckpt_path, cfg):

    vfe = PillarVFE(
        model_cfg=cfg.MODEL.VFE,
        num_point_features=4,
        point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE']
    )
    vfe.to('cuda').eval()

    checkpoint = torch.load(ckpt_path, map_location='cuda')
    dicts = {}
    for key in checkpoint['model_state'].keys():
        if "vfe" in key:
            dicts[key[4:]] = checkpoint['model_state'][key] # remove prefix "vfe."

    # dicts = fuse_conv_bn_weights(dicts)
    vfe.load_state_dict(dicts) 

    return vfe


def fuse_conv_bn_weights(dict, bn_eps=1e-3):
    conv_w,conv_b,bn_rm,bn_rv,bn_w ,bn_b=None,None,None,None,None,None
    for key,val in dict.items():
        if 'linear.weight' in key:
            conv_w= val
            conv_name = key
            bias_name = key.replace('weight','bias')
            continue
        if 'linear.bias' in key:
            conv_b= val
            continue
        if 'norm.bias' in key:
            bn_b= val
            continue
        if 'running_mean' in key:
            bn_rm= val
            continue

        if 'running_var' in key:
            bn_rv= val
            continue
        if 'norm.weight' in key:
            bn_w= val
            continue
        conv_b = val if 'linear.bias' in key else None

    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)

    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)

    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w.permute([1,0]) * (bn_w * bn_var_rsqrt)
    conv_w =conv_w.permute([1,0])
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return {conv_name:conv_w,bias_name:conv_b}

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/root/xiaohu/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--ckpt_path', type=str, default='/root/xiaohu/OpenPCDet/output/kitti_models/pointpillar/default/ckpt/pointpillar_7728.pth')
    parser.add_argument('--export_onnx_file', type=str, default='../model/pp_pfe.onnx')

    args = parser.parse_args()

    cfg_file = args.cfg_file
    ckpt_path = args.ckpt_path
    export_onnx_file = args.export_onnx_file

    print('Convert torch to onnx ... ')
    cfg_from_yaml_file(cfg_file, cfg)
    import pdb;

    pfe = build_vfe(ckpt_path, cfg)
    pfe.eval().cuda()
    # build input

    max_num_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS['test']
    max_points_per_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL
    dims_feature = pfe.num_point_features
    dummy_input = torch.ones(20000, max_points_per_pillars, dims_feature).cuda()
    
    torch.onnx.export(pfe,
                      dummy_input,
                      export_onnx_file,
                      opset_version=11,
                      verbose=True,
                      do_constant_folding=True,
                      )


    import onnx
    from onnxsim import simplify
    onnx_model = onnx.load(export_onnx_file)
    onnx_sim_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim_model, export_onnx_file)

