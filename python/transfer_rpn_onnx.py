import sys
sys.path.append("..")
from pcdet.config import cfg, cfg_from_yaml_file
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pcdet.models.dense_heads .anchor_head_template import AnchorHeadTemplate

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        self.res_backbone = self.model_cfg.get('res_backbone', False)

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3,stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                if self.res_backbone:
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01)
                    ])
                else:
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx],upsample_strides[idx],stride=upsample_strides[idx], bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],stride,stride=stride, bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            if self.res_backbone:
                x = self.blocks[i][:4](x)
                for mm in range(self.model_cfg.LAYER_NUMS[i]):
                    identity = x
                    out = self.blocks[i][4 + mm * 5:4 + (mm + 1) * 5](x)
                    x = x + out
            else:
                x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return x

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, x):
        data_dict={}
        spatial_features_2d = x

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None

        data_dict['batch_cls_preds']     = cls_preds
        data_dict['batch_box_preds']     = box_preds
        data_dict['cls_preds_normalized'] = dir_cls_preds

        return data_dict

class RPN(nn.Module):
    def __init__(self, cfg, grid_size):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
        # self.backbone_2d = MobileNetV2(cfg.MODEL.BACKBONE_2D,cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
        # self.backbone_2d = CSPDarknet53(cfg.MODEL.BACKBONE_2D, cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)

        print('The dense head is ', cfg.MODEL.DENSE_HEAD.NAME)
        if cfg.MODEL.DENSE_HEAD.NAME=='AnchorHeadMulti':
            self.dense_head = AnchorHeadMulti( model_cfg=cfg.MODEL.DENSE_HEAD,
                                               input_channels=384,
                                               num_class=len(cfg.CLASS_NAMES),
                                               class_names=cfg.CLASS_NAMES,
                                               grid_size=grid_size,
                                               point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                                               predict_boxes_when_training=False)
        elif cfg.MODEL.DENSE_HEAD.NAME=='AnchorHeadSingle':
            self.dense_head = AnchorHeadSingle(model_cfg=cfg.MODEL.DENSE_HEAD,
                                               input_channels=384,
                                               num_class=len(cfg.CLASS_NAMES),
                                               class_names=cfg.CLASS_NAMES,
                                               grid_size=grid_size,
                                               point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                                               predict_boxes_when_training=False)
        else:
            print('The densehead are not AnchorHeadMulti or AnchorHeadSingle !')
            sys.exit()

    def forward(self, spatial_features):
        x = self.backbone_2d(spatial_features)
        return self.dense_head.forward(x)

def fuse_conv_bn_weights(dit, bn_eps=1e-3):
    conv_w,conv_b,bn_rm,bn_rv,bn_w ,bn_b=None,None,None,None,None,None
    conv_w, bn_w,bn_b, bn_rm, bn_rv,_ = dit.values()
    conv_name = list(dit.keys())[0]
    bias_name =conv_name.replace('weight','bias')
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)

    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)

    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    if conv_name =='backbone_2d.deblocks.0.0.weight' or conv_name =='backbone_2d.deblocks.2.0.weight':
        conv_w = conv_w.permute([1,0,2,3])
    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1, 1, 1, 1])
    if conv_name =='backbone_2d.deblocks.0.0.weight' or conv_name =='backbone_2d.deblocks.2.0.weight':
        conv_w = conv_w.permute([1,0,2,3])
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return {conv_name:conv_w,bias_name:conv_b}

def merge_bn(dicts):
    dit={}
    new_dicts ={}
    for key,val in dicts.items():
        if len(dit)<6 and 'dense_head' not in key:
                dit[key]=val
                continue
        elif len(dit)==6 :
            # print(dit.keys())
            dt = fuse_conv_bn_weights(dit)
            for i,j in dt.items():
                new_dicts[i]=j
            dit = {}
            if 'dense_head'  in key:
                new_dicts[key] = val
                continue
            dit[key] = val

        else:
            new_dicts[key]=val
    return new_dicts

def build_rpn(ckpt_path, cfg):
    merge_bn =False
    point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    rpn = RPN(cfg, grid_size)
    rpn = rpn.eval()
    rpn.to('cuda')
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    dicts = {}
    for key in checkpoint["model_state"].keys():
        if 'backbone_2d' in key:
            dicts[key] = checkpoint["model_state"][key] # remove prefix 'backbone_2d.'
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
    if merge_bn:
        dicts =merge_bn(dicts)

    print("len(dicts)",len(dicts))
    rpn.load_state_dict(dicts)

    return rpn

def build_input(cfg):
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])
    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
    grid_x, grid_y = grid_size.astype(np.int32).tolist()[:2]
    dummpy_input = torch.ones(1,cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES, grid_y, grid_x).cuda()
    return dummpy_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--ckpt_path', type=str, default='output/kitti_models/pointpillar/default/ckpt/pointpillar_7728.pth')
    parser.add_argument('--export_onnx_file', type=str, default='../model/pp_rpn.onnx')
    args = parser.parse_args()

    ###################### single head #############################################
    cfg_file = args.cfg_file
    ckpt_path = args.ckpt_path
    export_onnx_file = args.export_onnx_file

    print('Convert torch to onnx ... ')
    # parse config
    cfg_from_yaml_file(cfg_file, cfg)
    # build network
    rpn = build_rpn(ckpt_path, cfg)
    # build input
    dummy_input = build_input(cfg)
    rpn.eval().cuda()

    torch.onnx.export(rpn,
                      dummy_input,
                      export_onnx_file,
                      opset_version=13,
                      verbose=True,
                      do_constant_folding=True)

    import onnx
    from onnxsim import simplify
    onnx_model = onnx.load(export_onnx_file)
    onnx_sim_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim_model, export_onnx_file)

