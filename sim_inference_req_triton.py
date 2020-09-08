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
import numpy as np
import os
import json
from builtins import range
import tensorrtserver.api.model_config_pb2 as model_config
from tensorrtserver.api import *
import time


def triton_inferer(ctx, input_name, output_name, batch):
    batch = [batch[i].astype(np.float32) for i in range(0,batch.shape[0]) ]
    
    input_dict = { input_name : batch }
    output_dict = { output_name : (InferContext.ResultFormat.RAW)}
    results = ctx.run(
        inputs=input_dict, 
        outputs=output_dict, 
        batch_size=80
    )
    return results[output_name]

def parse_model(url, protocol, model_name, batch_size, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements
    """
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    if len(config.input) != 1:
        raise Exception("expecting 1 input, got {}".format(len(config.input)))
    if len(config.output) != 1:
        raise Exception("expecting 1 output, got {}".format(len(config.output)))

    input = config.input[0]
    output = config.output[0]
    
    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).
    max_batch_size = config.max_batch_size
    if max_batch_size == 0:
        if batch_size != 1:
            raise Exception("batching not supported for model '" + model_name + "'")
    else: # max_batch_size > 0
        if batch_size > max_batch_size:
            raise Exception("expecting batch size <= {} for model {}".format(max_batch_size, model_name))

#     Model input must have 3 dims, either CHW or HWC
    if len(input.dims) != 3:
        raise Exception(
            "expecting input to have 3 dimensions, model '{}' input has {}".format(
                model_name, len(input.dims)))

    if input.format == model_config.ModelInput.FORMAT_NHWC:
        h = input.dims[0]
        w = input.dims[1]
        c = input.dims[2]
    else:
        c = input.dims[0]
        h = input.dims[1]
        w = input.dims[2]

    return (input.name, output.name, c, h, w, input.format, input.data_type)

def main():
    parser = argparse.ArgumentParser(description='Run inference with a model on the TRITON server')
    parser.add_argument('--model', help='Model in TRITON server')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')
    parser.add_argument('--interval', default=1, type=int,
                        metavar='N', help='interval to simulate inference requests')
    args = parser.parse_args()
    
    url = args.url
    protocol = ProtocolType.from_str(args.protocol)
    model_name = args.model
    model_version = -1
    
    try:
        print("Checking Health for model {}".format(model_name))
        health_ctx = ServerHealthContext(args.url, protocol,
                                     http_headers=args.http_headers)
        print("Live: {}".format(health_ctx.is_live()))
        print("Ready: {}".format(health_ctx.is_ready()))                                     
    except:
        raise RuntimeError("Model not available in server.. OR.. Is it running?")


    batch_size = 80
    input_name, output_name, c, h, w, format, dtype = parse_model(url, protocol, model_name, batch_size, verbose=True)

    ## Generate random inputs
    input_shape = (batch_size,c,h,w)
    inputs = np.random.random(input_shape).astype(np.float32)
    ctx = InferContext(url, protocol, model_name, model_version, verbose=False)
    
    request_interval = args.interval
    
    counter = 0
    while counter < 10000:
        out = triton_inferer(ctx, input_name, output_name, inputs)
        if counter % 100 == 0: print('Iteration {}'.format(counter))
        time.sleep(1)
        counter += 1


if __name__ == "__main__":
    main()
