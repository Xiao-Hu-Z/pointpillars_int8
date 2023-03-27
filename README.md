# Installation

Prepare the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) environment


# Preperation

## Export as onnx models

To export your own models, you can run

```bash
python3 export_onnx.py \
--cfg_file pointpillar.yaml
--ckpt_path your_model.pth \
--onnx_file pfe.onnx \

python3 export_onnx.py \
--cfg_file pointpillar.yaml
--ckpt_path your_model.pth \
--onnx_file rpn.onnx \
```

Here we extract two pure nn models from the whole computation graph `pfe` and `rpn`, this is to make it easier for trt to optimize its inference engines with int8.

## int8 quantization

### Generate calib_data

To make implicit ptq quantization, you need previously generate calibration files

The input of `backbone` directly modifies pcdet/models/detectors/pointpillar.py, and does not write code repeatedly

```python
    def forward(self, batch_dict):
        # for cur_module in self.module_list:
        #     batch_dict = cur_module(batch_dict)
        batch_dict = self.module_list[0](batch_dict)
        batch_dict = self.module_list[1](batch_dict)
        return batch_dict 
```

Run the following command to generate the calibration input file

```bash
python3 generate_calib_data.py \
--cfg_file pointpillar.yaml \
--data_path your dataset file \
--ckpt your_model.pth \
--calib_file_path store cal ibration input file
```

By default this will generate fp16-engine files.

### Generate TensorRT serialized engines

Actually you can directly create trt engines from onnx models and skip this step, however a more ideal way is to load your previously saved serialize engine files.

You can run 

```bash
python3 ptq_int8.py \
--config waymo_centerpoint_pp_two_pfn_stride1_3x.py \
--pfe_onnx_file rpn.onnx \
--rpn_onnx_file rpn.onnx \
--pfe_engine_path pfe_fp.engine \
--rpn_engine_path rpn_fp.engine \
--mode quantification mode in fp32, fp16 or int8 \
--calib_file_path  store cal ibration input file
```

By default this will generate int8-engine files.

You can also use the txtexec command to obtain fp16 or fp32's trt

```shell
# x86
# fp16
TensorRT-8.4.3.1/targets/x86_64-linux-gnu/bin/trtexec --onnx=pp_pfe.onnx --explicitBatch --saveEngine=pp_pfe_fp16_trtexec.trt --fp16 --workspace=1024 --verbose

# fp32
TensorRT-8.4.3.1/targets/x86_64-linux-gnu/bin/trtexec --onnx=pp_pfe.onnx --explicitBatch --saveEngine=pp_pfe_fp32_trtexec.trt --workspace=1024 --verbose
```

# Run inference
编译代码，会得到两个可执行文件，一个用于获取数据的推理结果，一个用于可视化

运行下列代码：
```
./test_point_pillars_cuda
```

对于单帧数据：data/000003.bin 结果输出：
```s
# pytorch :
8
16.999187 3.7984838 -0.94293684 4.421786 1.6759017 1.471789 6.259531 0.95879066 1
5.309882 11.392116 -1.2433618 3.5853221 1.5456746 1.4377103 3.157524 0.94660896 1
30.813366 -0.572901 -0.75715184 4.3241267 1.661644 1.5736197 6.2604256 0.91791165 1
53.294666 2.0207937 -0.40219212 3.9299285 1.6296755 1.5902154 6.171218 0.5074535 1
53.318325 -2.7356923 -0.4548468 3.872774 1.5616711 1.6064421 6.0431376 0.441119 1
18.733013 15.593122 -1.3264668 4.466153 1.6951573 1.5360177 6.388626 0.32318175 1
0.6559625 -0.5785154 -1.131638 4.1749973 1.6668894 1.560942 6.034973 0.17306222 1
64.31819 8.491507 -0.48887318 4.0612826 1.5959842 1.5436162 6.276266 0.14615808 1

# c++/cuda/tensorrt
## fp32
8
16.999060 3.798431 -0.942883 4.421756 1.675821 1.471751 6.259527 0.958724 0.000000 3.798431 1 
5.309843 11.392059 -1.243091 3.585154 1.545641 1.437766 3.157501 0.946587 0.000000 11.392059 1 
30.813334 -0.572902 -0.757078 4.324007 1.661577 1.573622 6.260392 0.917729 0.000000 -0.572902 1 
53.294670 2.020855 -0.402193 3.929884 1.629674 1.590329 6.171206 0.507371 0.000000 2.020855 1 
53.318218 -2.735692 -0.454844 3.872712 1.561591 1.606453 6.043070 0.440806 0.000000 -2.735692 1 
18.732822 15.593171 -1.326382 4.465864 1.695055 1.535907 6.388664 0.322872 0.000000 15.593171 1 
0.650379 -0.578194 -1.131110 4.173248 1.666790 1.560607 6.036040 0.174399 0.000000 -0.578194 1 
64.318169 8.491416 -0.488864 4.061276 1.596020 1.543714 6.276298 0.146148 0.000000 8.491416 1

## fp16
8
16.998623 3.798585 -0.942919 4.422517 1.675865 1.471489 6.259427 0.958924 0.000000 3.798585 1 
5.310803 11.391916 -1.243179 3.584088 1.545765 1.437490 3.157401 0.947088 0.000000 11.391916 1 
30.812458 -0.572833 -0.757393 4.321916 1.660997 1.572475 6.260312 0.917746 0.000000 -0.572833 1 
53.294903 2.017216 -0.407383 3.933108 1.630412 1.597092 6.169904 0.569374 0.000000 2.017216 1 
53.315487 -2.735236 -0.455371 3.879523 1.567689 1.614882 6.045636 0.475118 0.000000 -2.735236 1 
18.732012 15.592730 -1.323921 4.463737 1.694481 1.535416 6.388288 0.321673 0.000000 15.592730 1 
64.321220 8.491755 -0.483936 4.050905 1.591818 1.535908 6.274869 0.185652 0.000000 8.491755 1 
0.646314 -0.577143 -1.130635 4.173886 1.667091 1.560809 6.028302 0.175538 0.000000 -0.577143 1
```


# eval

将预测数据核openpcdet得到pkl文件转为数据评估格式

```bash
cd /eval
python kitti_format.py
```
数据存储在 kitti/object/pred/ 目录下

- Run evaluation kit on prediction and pcdet outputs

> 参考：https://github.com/traveller59/kitti-object-eval-python，我把相应的的依赖函数提取出放到kitti-object-eval-python里了，不需要单独安装second-1.5.1，spconv-1.0

```python
python ./kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti/object/training/label_2/ --result_path=./kitti/object/pcdet/ --label_split_file=./val.txt --current_class=0,1,2 --coco=False

Car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:90.77, 89.77, 88.75
bev  AP:89.52, 87.06, 84.11
3d   AP:85.96, 77.09, 74.41
aos  AP:90.76, 89.57, 88.42
Car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:90.77, 89.77, 88.75
bev  AP:90.78, 90.15, 89.42
3d   AP:90.78, 90.03, 89.19
aos  AP:90.76, 89.57, 88.42
Pedestrian AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:66.18, 62.16, 59.18
bev  AP:61.34, 56.08, 52.51
3d   AP:56.59, 51.94, 47.61
aos  AP:48.25, 45.33, 42.83
Pedestrian AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:66.18, 62.16, 59.18
bev  AP:72.08, 69.11, 66.13
3d   AP:72.00, 68.88, 64.93
aos  AP:48.25, 45.33, 42.83
Cyclist AP(Average Precision)@0.50, 0.50, 0.50:
bbox AP:85.03, 72.54, 68.55
bev  AP:81.97, 66.16, 62.32
3d   AP:79.70, 62.35, 59.36
aos  AP:84.48, 70.69, 66.66
Cyclist AP(Average Precision)@0.50, 0.25, 0.25:
bbox AP:85.03, 72.54, 68.55
bev  AP:86.24, 70.33, 66.47
3d   AP:86.24, 70.33, 66.47
```

计算TensorRT推理结果：
```
python ./kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti/object/training/label_2/ --result_path=./kitti/object/pred/fp32 --label_split_file=./val.txt --current_class=0,1,2 --coco=False

python ./kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti/object/training/label_2/ --result_path=./kitti/object/pred/fp16 --label_split_file=./val.txt --current_class=0,1,2 --coco=False

python ./kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti/object/training/label_2/ --result_path=./kitti/object/pred/int8 --label_split_file=./val.txt --current_class=0,1,2 --coco=False
```

# Speed analysis

| Model               | 3060ti |
| ------------------- | ------ |
| PointPillars-FP32   | 8.06   |
| PointPillars-FP16   | 5.15   |
| PointPillars-int8   | 4.24   |
| pfe_FP32 + rpn_fp16 | 5.95   |
| pfe_FP16 + rpn_fp32 | 7.30   |

# Metrics analysis

在Car、Pedestrian、Cyclist交并比分别为0.7、0.5、0.5，中等难度val数据集的3D检测性能如下：



| Model             | Car@R11 | Pedestrian@R11 | Cyclist@R11 |
| ----------------- | ------- | -------------- | ----------- |
| OpenPCDet         | 77.09   | 51.94          | 62.35       |
| PointPillars-FP32 | 74.75   | 52.82          | 60.89       |
| PointPillars-FP16 | 74.70   | 52.78          | 60.94       |
| PointPillars-int8 | 60.53   | 10.79          | 7.57        |

也可以混合精度测试，通过修改config里的yaml参数，测试评估时要保证路径一一对应

可以看出int8精度损失比较严重，需要进一步做感知训练量化，可以利用英伟达提供的[量化工具箱](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)进一步做感知训练量化



# 可视化

编译好代码，ros播放kitti的bag包，在build目录下执行以下代码：
```
./test_point_pillars_cuda_ros
```



# Acknowledgements

This project refers to some codes from:

[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

[TensorRT](https://github.com/NVIDIA/TensorRT/tree/master)
