import os
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/root/xiaohu/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/groupdata/share/openset/kitti/training/velodyne/',help='specify the point cloud data file or directory')
    #parser.add_argument('--data_path', type=str, default='/root/xiaohu/OpenPCDet/cpp/point_pillars_kitti/data/000000.bin',help='specify the point cloud data file or directory')

    parser.add_argument('--ckpt', type=str, default="/root/xiaohu/OpenPCDet/output/kitti_models/pointpillar/default/ckpt/pointpillar_7728.pth", help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument("--calib_file_path", help="the dir to calibration files, only config when `quant` is enabled. ",type = str, default='../data/calib_data/training2/')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

args, cfg = parse_config()

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel
    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        # if self.model_cfg.WITH_NUM_POINTS:
        #     num_point_features += 1
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

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def forward(self, voxel_features, voxel_num_points, coords):
        features_ls = [voxel_features]
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        device = voxel_features.device
        # 使用矩阵切片代替取下标方式,flip按照维度对输入旋转z,y,x
        f_center = voxel_features[..., :3] - (coords[..., 1:] * torch.tensor([self.voxel_z, self.voxel_y, self.voxel_x]).to(device) + torch.tensor([self.z_offset, self.y_offset, self.x_offset]).to(device)).unsqueeze(1).flip(2)
        features_ls.append(f_center)
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1] # 32
        mask = get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()

        return features

def pad_voxel(cfg,voxel_features, voxel_num_points, coords, max_pillar_num = None):
    point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)

    voxel_x = voxel_size[0]
    voxel_y = voxel_size[1]
    voxel_z = voxel_size[2]
    x_offset = voxel_x / 2 + point_cloud_range[0]
    y_offset = voxel_y / 2 + point_cloud_range[1]
    z_offset = voxel_z / 2 + point_cloud_range[2]
    _with_distance = False

    points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    f_cluster = voxel_features[:, :, :3] - points_mean
    f_center = torch.zeros_like(voxel_features[:, :, :3])
    f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * voxel_x + x_offset)
    f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * voxel_y + y_offset)
    f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * voxel_z + z_offset)

    # Combine together feature decorations
    features = [voxel_features, f_cluster, f_center]
    if _with_distance:
        points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
        features.append(points_dist)
    features = torch.cat(features, dim=-1)

    # 下面方式生成的pfe_input 数据类型不一致，导致读取pfe_bin 后合并一维后维度变大（2倍）
    # features_ls = [voxel_features]
    # points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    # f_cluster = voxel_features[:, :, :3] - points_mean
    # features_ls.append(f_cluster)

    # device = voxel_features.device 
    # f_center = voxel_features[..., :3] - (coords[..., 1:] * torch.tensor([voxel_z, voxel_y, voxel_x]).to(device) + torch.tensor([z_offset, y_offset, x_offset]).to(device)).unsqueeze(1).flip(2)
    # features_ls.append(f_center)
    # features = torch.cat(features_ls, dim=-1)
    # The feature decorations were calculated without regard to whether
    # pillar was empty. Need to ensure that
    # empty pillars remain set to zeros.
    voxel_count = features.shape[1] # 32
    mask = get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask

    # onnx是定长的，多余需要补0
    if max_pillar_num is not None:
        pillar_size  = [x for x in features.shape]
        if max_pillar_num < pillar_size[0]:
            features = features[:max_pillar_num]
        else:
            pillar_size[0] = max_pillar_num - pillar_size[0]
            zeros = torch.zeros(pillar_size).to(features.device)
            features = torch.cat([features, zeros],axis = 0)
    return features

def convert2scatter(inputs,indexs):
    assert len(inputs.shape) == 2
#     assert len(inputs) == len(indexs)
    dim = inputs.shape[-1]
    rets = torch.zeros((dim,496,432),dtype=inputs.dtype).to(inputs.device)
    num_pillars = min(len(indexs), 40000)
    indexs = indexs.type(torch.long)
    for i in range(num_pillars):
        if indexs[i] <0 or indexs[i] >= 432 * 496: continue
        yIdx = indexs[i] // 432
        xIdx = indexs[i] %  432
        rets[:,yIdx,xIdx] = inputs[i]
    return rets

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, pillar_features, coords):
        spatial_feature = torch.zeros(self.num_bev_features, 
                                        self.nx * self.ny, 
                                        dtype=pillar_features.dtype, 
                                        device=pillar_features.device)
        indices =  coords[:, 2] * self.nx + coords[:, 3] #432 * y + x
        indices = indices.long()
        pillars_feature = pillar_features.t()  # float32[64,20000]
        spatial_feature[:, indices] = pillars_feature
        spatial_feature = spatial_feature.view(1,self.num_bev_features, self.ny, self.nx) # 对应onnx resahap


        return spatial_feature

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            # (43400, 5)
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def main():
    
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    vfe = PillarVFE(
        model_cfg=cfg.MODEL.VFE,
        num_point_features=cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES'],
        point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])

    vfe.to('cuda').eval()

    point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    scatter = PointPillarScatter(
        model_cfg=cfg.MODEL.MAP_TO_BEV,
        grid_size= grid_size,
        voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])

    scatter.to('cuda').eval()
    with torch.no_grad():
        for idx, data_dict in tqdm(enumerate(demo_dataset)):
            if idx > 1999:
                break
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            # 注意参数要与c++
            # pfe_inputs = pad_voxel(cfg, data_dict["voxels"], data_dict["voxel_num_points"], data_dict["voxel_coords"],max_pillar_num = 40000)
            # pfe_inputs = pfe_inputs.cpu().numpy()
            # pfe_inputs.tofile(os.path.join(args.calib_file_path , str(idx) + "_pfe_input.bin"))
            # np.savetxt('pfe_inputs.txt',pfe_inputs.reshape(40000,32*10),fmt='%.5f', delimiter=' ')
            batch_dict = model.forward(data_dict)
            pfe_outputs = batch_dict['pillar_features']
            # indices =  coords[:, 2] * 432 + coords[:, 3] # 432 * y + x
            # batch_id,z,y,x
            indices = data_dict["voxel_coords"][:, 2] * 432 + data_dict["voxel_coords"][:, 3]
            scatter_outputs = convert2scatter(pfe_outputs,indices).cpu().numpy()
            scatter_outputs.tofile(os.path.join(args.calib_file_path , str(idx) + "_rpn_input.bin"))
            # rpn_input.tofile(os.path.join(args.calib_file_path , str(idx) + "_rpn_input.bin"))
            # rpn_input = batch_dict['spatial_features'].cpu().numpy()
            # rpn_input.tofile(os.path.join(args.calib_file_path , str(idx) + "_rpn_input.bin"))

            # np.savetxt('scatter_outputs.txt',scatter_outputs.reshape(64,496*432),fmt='%.5f', delimiter=' ')
            # np.savetxt('rpn_input.txt',rpn_input.reshape(64,496*432),fmt='%.5f', delimiter=' ')
            # mse = np.mean(scatter_outputs- rpn_input)
            # print("mse:",mse)


if __name__ == '__main__':
    main()


