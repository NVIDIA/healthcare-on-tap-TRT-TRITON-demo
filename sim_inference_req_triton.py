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

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
import time
import sys

def triton_inferer(triton_client, model_name, input_name, output_name, batch, headers=None):

    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput(input_name, batch.shape, "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(batch, binary_data=True)
    outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))
    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(model_name,
                                  inputs,
                                  outputs=outputs,
                                  query_params=query_params,
                                  headers=headers)
    return results.as_numpy(output_name)

def parse_model(model_metadata, model_config, batch_size):
    """
    Check the configuration of a model to make sure it meets the
    requirements
    """
    if len(model_config['input']) != 1:
        raise Exception("expecting 1 input, got {}".format(len(config.input)))
    # if len(config.output) != 1:
    #     raise Exception("expecting 1 output, got {}".format(len(config.output)))

    input_metadata = model_metadata['inputs'][0]
    input_config = model_config['input'][0]
    output_metadata = model_metadata['outputs'][0]

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']
    
    
    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).

    if max_batch_size == 0:
        if batch_size != 1:
            raise Exception("batching not supported for model '" + model_metadata['name'] + "'")
    else: # max_batch_size > 0
        if batch_size > max_batch_size:
            raise Exception("expecting batch size <= {} for model {}".format(max_batch_size, model_metadata['name']))

#     Model input must have 3 dims, either CHW or HWC
    if len(input_metadata['shape']) - 1 != 3:
        raise Exception(
            "expecting input to have 3 dimensions, model '{}' input has {}".format(
                model_metadata['name'], len(input_metadata['shape'])))

    input_batch_dim = (max_batch_size > 0)
    if input_config['format'] == "FORMAT_NHWC":
        h = input_metadata['shape'][1 if input_batch_dim else 0]
        w = input_metadata['shape'][2 if input_batch_dim else 1]
        c = input_metadata['shape'][3 if input_batch_dim else 2]
    else:
        c = input_metadata['shape'][1 if input_batch_dim else 0]
        h = input_metadata['shape'][2 if input_batch_dim else 1]
        w = input_metadata['shape'][3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata['name'], output_metadata['name'], c,
            h, w, input_config['format'], input_metadata['datatype'])

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
    # protocol = ProtocolType.from_str(args.protocol)
    model_name = args.model
    # model_version = -1
    
    triton_client = httpclient.InferenceServerClient(
                url=args.url)
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    batch_size = 80
    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(model_metadata, model_config, batch_size)

    ## Generate random inputs
    input_shape = (batch_size,c,h,w)
    inputs = np.random.random(input_shape).astype(np.float32)
    # ctx = InferContext(url, protocol, model_name, model_version, verbose=False)
    
    request_interval = args.interval
    
    counter = 0
    while counter < 10000:
        out = triton_inferer(triton_client, model_name, input_name, output_name, inputs)
        if counter % 100 == 0: print('Iteration {}'.format(counter))
        time.sleep(1)
        counter += 1


if __name__ == "__main__":
    main()
