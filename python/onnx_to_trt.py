#############################################
# python api转
#############################################
import sys
import os
import argparse
import tensorrt as trt

import faulthandler
faulthandler.enable()

# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
#     parser.add_argument("--onnx_path", type=str, default='../model/pp_pfe.onnx')
#     parser.add_argument("--trt_path", type=str, default='../model/pp_pfe_fp32_2.enigen')
#     args = parser.parse_args()
#     onnx_file_path = args.onnx_path
#     engine_file_path = args.trt_path
#     print('get start')
#     TRT_LOGGER = trt.Logger()
#     with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#         config = builder.create_builder_config()
#         config.max_workspace_size =( 1 << 30 ) * 2 # 2 GB
#         builder.max_batch_size = 1
#         # config.set_flag(trt.BuilderFlag.FP16)
#         # Parse model file
#         print('Loading ONNX file from path {}...'.format(onnx_file_path))
#         with open(onnx_file_path, 'rb') as model:
#             print('Beginning ONNX file parsing')
#             if not parser.parse(model.read()):
#                 print ('ERROR: Failed to parse the ONNX file.')
#                 for error in range(parser.num_errors):
#                     print (parser.get_error(error))
        
#         print(f"raw shape of {network.get_input(0).name} is: ", network.get_input(0).shape)
#         # print(f"raw shape of {network.get_input(1).name} is: ", network.get_input(1).shape)
#         # print(f"raw shape of {network.get_input(2).name} is: ", network.get_input(2).shape)


#         print('Completed parsing of ONNX file')
#         print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
#         # engine = builder.build_engine(network,config)
#         engine = builder.build_serialized_network(network,config)
#         print("Completed creating Engine")
#         with open(engine_file_path, "wb") as f:
#             # f.write(engine.serialize())
#             f.write(engine)

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def get_engine(onnx_file_path, engine_file_path):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        builder = trt.Builder(TRT_LOGGER)
        # Set max threads that can be used by builder.
        builder.max_threads = 10
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        runtime = trt.Runtime(TRT_LOGGER)
        # Set max threads that can be used by runtime.
        runtime.max_threads = 10

        # Parse model file
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("Completed parsing of ONNX file")

        # Print input info
        print("Network inputs:")
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        # network.get_input(0).shape = [10, 1]
        # network.get_input(1).shape = [10, 1, 1, 16]
        # network.get_input(2).shape = [6, 1]
        # network.get_input(3).shape = [6, 1, 1, 16]

        config = builder.create_builder_config()
        # config.set_flag(trt.BuilderFlag.FP16)
        # config.set_flag(trt.BuilderFlag.REFIT)
        #  DeprecationWarning
        # config.max_workspace_size = 1 << 28  # 256MiB
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28) 

        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            runtime.max_threads = 10
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--onnx_path", type=str, default='../model/pp_pfe.onnx')
    parser.add_argument("--trt_path", type=str, default='../model/pp_pfe_fp32_4.enigen')
    args = parser.parse_args()
    onnx_file_path = args.onnx_path
    engine_file_path = args.trt_path
    get_engine(onnx_file_path,engine_file_path)

    with open(engine_file_path, "rb") as f:
        serialized_engine = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()








