# pip install pycuda -i https://pypi.tuna.tsinghua.edu.cn/simple
import glob
import torch
import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import matplotlib.pyplot as plt
from time import time
import argparse 

class MinMaxCalibrator(trt.IInt8MinMaxCalibrator): 
    def __init__(self, datas, cache_file="calib_cache.bin", batch_size=1,shape = [30000,32,10]):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.shape = shape
        self.shape.insert(0,batch_size)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        assert isinstance(datas, list) and len(datas), "datas should be a type of `list`and should not be empty. "
        # if isinstance(datas[0], str) :   self.read_cache = False
        # elif isinstance(datas[0], np.ndarray) : self.read_cache = True
        # else: raise TypeError("Can't recognize calibration data types.")
        self.datas = datas
        self.data = self.read_data(0)

        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        # print("alloc size " , self.data[0].nbytes * self.batch_size)
        self.device_input = cuda.mem_alloc(self.data.nbytes * self.batch_size)
    def read_data(self,idx):
        data = np.fromfile(self.datas[idx],dtype = np.float32)
        return data
        
    def get_batch_size(self):
        return self.batch_size
    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.datas):
            return None
        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}/{}".format(current_batch,  len(self.datas)//self.batch_size ))
        batch = []
        for i in range(self.batch_size):
            sample = self.read_data(self.current_index + i)
            batch.append(sample)
        batch = np.stack(batch,axis = 0)
        batch = np.ascontiguousarray(batch.reshape(*self.shape)).ravel()
        #batch = self.read_data(self.current_index )
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()
        pass
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)



class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    
    def __init__(self, datas, cache_file="calib_cache.bin", batch_size=1,shape = [30000,32,10]):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        # 使用输入节点名称和批处理流创建一个IInt8EntropyCalibrator2对象
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.shape = shape 
        self.shape.insert(0,batch_size) # [1,30000,32,10]
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        assert isinstance(datas, list) and len(datas), "datas should be a type of `list`and should not be empty. "
        # if isinstance(datas[0], str) :   self.read_cache = False
        # elif isinstance(datas[0], np.ndarray) : self.read_cache = True
        # else: raise TypeError("Can't recognize calibration data types.")
        self.datas = datas
        self.data = self.read_data(0)

        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        # print("alloc size " , self.data[0].nbytes * self.batch_size)
        # # 在设备上申请存储空间
        self.device_input = cuda.mem_alloc(self.data.nbytes * self.batch_size)
    def read_data(self,idx):
        data = np.fromfile(self.datas[idx],dtype = np.float32)
        return data

    # 获取batch大小 
    def get_batch_size(self):
        return self.batch_size
    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    # TensorRT将引擎绑定的名称传递给get_batch函数。您不必使用它们，但它们可以帮助您理解输入的顺序。绑定列表的顺序应与name相同。
    # 获取一个batch的数据
    # 只需要根据你的数据集存放路径及格式，读取一个batch即可。需要注意的是，读取的一个batch数据，数据类型是np.ndarray，shape为[batch_size, C, H, W]，也即[batch大小, 通道, 高, 宽]
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.datas):
            return None
        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}/{}".format(current_batch,  len(self.datas)//self.batch_size ))
        batch = []
        for i in range(self.batch_size):
            sample = self.read_data(self.current_index + i)
            batch.append(sample)
        # axis为0，表示它堆叠方向为第0维，堆叠的内容为数组第0维的数据
        batch = np.stack(batch,axis = 0)
        # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        # avel()方法将数组维度拉成一维数组
        batch = np.ascontiguousarray(batch.reshape(*self.shape)).ravel()
        #batch = self.read_data(self.current_index )
        # # 将数组从host拷贝到设备
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]
    # 将校准集写入缓存
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()
        pass
    # 从缓存读出校准集
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="TensorRT int8 quantization args")
    parser.add_argument("--pfe_onnx_path", help="the dir to load pfe  onnx",type = str, default = "../model/pp_pfe.onnx")
    parser.add_argument("--rpn_onnx_path", help="the dir to load rpn  onnx",type = str, default = "../model/pp_rpn.onnx")
    parser.add_argument("--pfe_engine_path", help="the dir to save pfe  engine",type = str, default = "../model/minmax_calib/pp_pfe_int8_python_training2.trt")
    parser.add_argument("--rpn_engine_path", help="the dir to save rpn  engine",type = str, default = "../model/minmax_calib/pp_rpn_int8_python_training2.trt")
    parser.add_argument("--mode", type=str, default='int8', help='fp32, fp16 or int8')
    parser.add_argument("--minmax_calib", default= 'true', help='whether to make MinMaxCalibration, by default we use EntropyCalib! ')
    parser.add_argument("--calib_file_path",default='../data/calib_data/training', help="the dir to calibration files, only config when `quant` is enabled. ",type = str)
    parser.add_argument("--calib_batch_size", type = int , default = 1, help = "batch size for calibration.")
    args = parser.parse_args()

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"
    if args.mode.lower() == 'int8':
        # lines = txt_file.readlines()
        # np.random.shuffle(lines)
        # calib_files = [os.path.join(args.calib_file_path,line.rstrip() + '.bin') for line in lines]
        pfe_calib_files = glob.glob(os.path.join(args.calib_file_path, "*pfe_input.bin") )
        rpn_calib_files = glob.glob(os.path.join(args.calib_file_path, "*rpn_input.bin") )
        print("%d calib files for each model. " % (len(rpn_calib_files)) )
        np.random.shuffle(pfe_calib_files)
        np.random.shuffle(rpn_calib_files)

    # for engine creation 
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # # create_builder_config 用于设置网络的最大工作空间等参数
    # # EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # with trt.Builder(TRT_LOGGER) as builder, \
    # builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
    # trt.OnnxParser(network,TRT_LOGGER) as parser, \
    # builder.create_builder_config() as config:
    #     # config.max_workspace_size = 1 << 30
    #     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28) 
    #     # IR 转换时，如果有多 Batch、多输入、动态 shape 的需求，都可以通过多次调用 set_shape 函数进行设置
    #     # set_shape 函数接受的传参分别是：输入节点名称，可接受的最小输入尺寸，最优的输入尺寸，可接受的最大输入尺寸。一般要求这三个尺寸的大小关系为单调递增
    #     if args.mode.lower() == 'int8':
    #         # 设置INT8模式和INT8校准器
    #         config.set_flag(trt.BuilderFlag.INT8)
    #         if args.minmax_calib:
    #             # CalibrationTable文件是在构建阶段运行校准算法时生成的。创建校准文件后，可以读取该文件以进行后续运行，而无需再次运行校准。
    #             pfe_calib = MinMaxCalibrator(pfe_calib_files,  cache_file="pfe_calib_cache_training.bin", batch_size = args.calib_batch_size, shape = [30000, 32, 10] )
    #         else:
    #             # 使用输入节点名称和批处理流创建一个Int8_calibrator对象
    #             pfe_calib = EntropyCalibrator(pfe_calib_files,  cache_file="pfe_calib_cache_training.bin", batch_size = args.calib_batch_size, shape = [30000, 32, 10] )
    #         config.int8_calibrator = pfe_calib
    #     elif args.mode.lower() == 'fp16':
    #         config.set_flag(trt.BuilderFlag.FP16)

    #     print('Loading pfe ONNX file from path {}...'.format(args.pfe_onnx_path))
    #     parsed = parser.parse_from_file(args.pfe_onnx_path)
    #     if parsed:
    #         print("building pfe trt engine . . .")
    #         serialized_engine = builder.build_serialized_network(network,config)
    #         with open(args.pfe_engine_path, 'wb') as f:
    #             f.write(serialized_engine)

    #         # print("deserialize the pfe engine . . . ")
    #         # runtime = trt.Runtime(TRT_LOGGER)
    #         # engine = runtime.deserialize_cuda_engine(serialized_engine)
    #         # context = engine.create_execution_context()
    #         # print("context_pfe", context)
    #     else:
    #         print("Parsing Failed ! ")
    #         for i in range(parser.num_errors):
    #             print(parser.get_error(i))

    # for rpn
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, \
    builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
    trt.OnnxParser(network,TRT_LOGGER) as parser, \
    builder.create_builder_config() as config:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 
        # config.max_workspace_size = 1 << 30
        builder.max_batch_size = 1
        if args.mode.lower() == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            if args.minmax_calib:
                rpn_calib = MinMaxCalibrator(rpn_calib_files,  cache_file="rpn_calib_cache_training.bin", batch_size = args.calib_batch_size, shape = [64, 496, 432] )
            else:
                rpn_calib = EntropyCalibrator(rpn_calib_files,  cache_file="rpn_calib_cache_training.bin", batch_size = args.calib_batch_size, shape = [64, 496, 432] )
            config.int8_calibrator = rpn_calib
        elif args.mode.lower() == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)

        print('Loading rpn ONNX file from path {}...'.format(args.rpn_onnx_path))
        parsed = parser.parse_from_file(args.rpn_onnx_path)
        if parsed:
            print("building rpn trt engine . . .")
            serialized_engine = builder.build_serialized_network(network,config)
            with open(args.rpn_engine_path, 'wb') as f:
                f.write(serialized_engine)

        else:
            print("Parsing Failed ! ")
            for i in range(parser.num_errors):
                print(parser.get_error(i))

    # # for engine creation 
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # # create_builder_config 用于设置网络的最大工作空间等参数
    # # EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # with trt.Builder(TRT_LOGGER) as builder, \
    # builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
    # trt.OnnxParser(network,TRT_LOGGER) as parser, \
    # builder.create_builder_config() as config:
    #     # config.max_workspace_size = 1 << 30
    #     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 
    #     # IR 转换时，如果有多 Batch、多输入、动态 shape 的需求，都可以通过多次调用 set_shape 函数进行设置
    #     # set_shape 函数接受的传参分别是：输入节点名称，可接受的最小输入尺寸，最优的输入尺寸，可接受的最大输入尺寸。一般要求这三个尺寸的大小关系为单调递增
    #     if args.mode.lower() == 'int8':
    #         # 设置INT8模式和INT8校准器
    #         config.set_flag(trt.BuilderFlag.INT8)
    #         if args.minmax_calib:
    #             # CalibrationTable文件是在构建阶段运行校准算法时生成的。创建校准文件后，可以读取该文件以进行后续运行，而无需再次运行校准。
    #             rpn_calib = MinMaxCalibrator(rpn_calib_files,  cache_file="rpn_calib_cache.bin", batch_size = args.calib_batch_size, shape = [64, 496, 432] )
    #         else:
    #             # 使用输入节点名称和批处理流创建一个Int8_calibrator对象
    #             rpn_calib = EntropyCalibrator(rpn_calib_files,  cache_file="rpn_calib_cache.bin", batch_size = args.calib_batch_size, shape = [64, 496, 432] )
    #         config.int8_calibrator = rpn_calib
    #     elif args.mode.lower() == 'fp16':
    #         config.set_flag(trt.BuilderFlag.FP16)

    #     print('Loading rpn ONNX file from path {}...'.format(args.rpn_onnx_path))
    #     parsed = parser.parse_from_file(args.rpn_onnx_path)
    #     if parsed:
    #         print("building rpn trt engine . . .")
    #         serialized_engine = builder.build_serialized_network(network,config)
    #         with open(args.rpn_engine_path, 'wb') as f:
    #             f.write(serialized_engine)

    #         # print("deserialize the rpn engine . . . ")
    #         # runtime = trt.Runtime(TRT_LOGGER)
    #         # engine = runtime.deserialize_cuda_engine(serialized_engine)
    #         # context = engine.create_execution_context()
    #         # print("context_pfe", context)
    #     else:
    #         print("Parsing Failed ! ")
    #         for i in range(parser.num_errors):
    #             print(parser.get_error(i))
# python ptq_int8.py --mode int8 
# python ptq_int8.py --mode int8