"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import argparse
import tensorrt as trt

def build_engine(args):

    print('Loading custom plugins')
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    print('Building TRT engine from ONNX file:', args.model)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = 8 << 30

        # Specifying runtime Dimensions
        # 
        # profile = builder.create_optimization_profile()
        # profile.set_shape("input", (1 , 3, 256, 256), (4 , 3, 256, 256), (8 , 3, 256, 256))
        # config = builder.create_builder_config()
        # config.add_optimization_profile(profile) 
        
        if args.fp16:
            print('Using FP16')
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)    # probably don't need/want this flag anywa 

        with open(args.model, 'rb') as model:
            if not parser.parse(model.read()):
                print('Throwing an Error')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
            else:
                print('Parsing Model')
                engine = builder.build_engine(network=network, config=config)
                print(engine)
                return engine

def save_engine(engine, engine_dest_path):
    print('Saving the engine file at:', engine_dest_path)
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def main():
    parser = argparse.ArgumentParser(description='Convert onnx model to TensorRT engine.')
    parser.add_argument('--model', help='The path to the .onnx model file')
    parser.add_argument('--output', default='.', help='The path save the .engine file')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Do conversion in fp16 mode.')
    args = parser.parse_args()

    engine = build_engine(args)
    if engine is not None:
        save_engine(engine, args.output)
    else:
        print('something is wrong')

if __name__ == '__main__':
    main()
