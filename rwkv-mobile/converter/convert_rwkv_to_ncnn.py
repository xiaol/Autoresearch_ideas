import argparse
import torch
import numpy as np

from ncnn_model_utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('input', help='input rwkv model file')
argparser.add_argument('output_name', help='output ncnn model name')
argparser.add_argument('--datatype', choices=['fp32', 'fp16'], default='fp16', help='model data type')
args = argparser.parse_args()

weights = torch.load(args.input, map_location='cpu')

version, n_layer, n_head, head_size, vocab_size = check_rwkv_info(weights)
print('version:', version)
print('n_layer:', n_layer)
print('n_head:', n_head)
print('head_size:', head_size)
print('vocab_size:', vocab_size)

ncnn_weights_file = open(args.output_name + '.bin', 'wb')
use_fp32 = args.datatype == 'fp32'

layer_count = 0
blob_count = 0
ncnn_param_lines = ['7767517\n', '[layer_count] [blob_count]\n']
layer_count, blob_count = build_info(ncnn_param_lines, ncnn_weights_file, version, n_layer, n_head, head_size, vocab_size, layer_count, blob_count)

layer_count, blob_count = build_inp_emb(ncnn_param_lines, ncnn_weights_file, weights, n_head, head_size, vocab_size, n_layer, layer_count, blob_count, use_fp32=use_fp32)

layer_input = 'emb'
for i in range(n_layer):
    if version == 6:
        layer_count, blob_count = build_time_mixing_v6(ncnn_param_lines, ncnn_weights_file, weights, layer_input, f'time_mixing_{i}_out', i, layer_count, blob_count, n_head, head_size, use_fp32=use_fp32)
        layer_count, blob_count = build_channel_mixing_v6(ncnn_param_lines, ncnn_weights_file, weights, f'time_mixing_{i}_out', f'channel_mixing_{i}_out', i, layer_count, blob_count, use_fp32=use_fp32)
    elif version == 7:
        layer_count, blob_count = build_time_mixing_v7(ncnn_param_lines, ncnn_weights_file, weights, layer_input, f'time_mixing_{i}_out', i, layer_count, blob_count, n_head, head_size, use_fp32=use_fp32)
        layer_count, blob_count = build_channel_mixing_v7(ncnn_param_lines, ncnn_weights_file, weights, f'time_mixing_{i}_out', f'channel_mixing_{i}_out', i, layer_count, blob_count, use_fp32=use_fp32)
    else:
        assert 0, f'unsupported version {version}'
    layer_input = f'channel_mixing_{i}_out'
layer_count, blob_count = build_output_head(ncnn_param_lines, ncnn_weights_file, weights, layer_input, 'logits', layer_count, blob_count, use_fp32=use_fp32)

ncnn_param_lines[1] = f'{layer_count} {blob_count}\n'
with open(args.output_name + '.param', 'w') as ncnn_param_file:
    ncnn_param_file.writelines(ncnn_param_lines)

ncnn_weights_file.close()
