from rwkv_src.rwkv_modeling import RWKV_RNN
from rwkv_src.model_utils import get_dummy_input_for_rwkv_causal_llm
from pathlib import Path
import argparse, types, os
import torch
import onnx
from torch.onnx import register_custom_op_symbolic

argparser = argparse.ArgumentParser()
argparser.add_argument('input', help='input rwkv model file')
argparser.add_argument('output_name', help='output onnx model name')
argparser.add_argument('--datatype', choices=['fp32', 'fp16'], default='fp32', help='model data type')
args = argparser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = True if args.datatype == 'fp16' else False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.USE_ONNX_REDUCE_L2 = True
model_args.USE_ONNX_L2NORM = False

model_args.MODEL_NAME = str(args.input).replace('.pth', '')
model = RWKV_RNN(model_args)

def norm(g, self):
    return g.op("LpNormalization", self, p_i=2, axis_i=-1)

def reducel2(g, self):
    return g.op("ReduceL2", self, axes_i=[-1])

register_custom_op_symbolic('customop::l2norm', norm, 4)
register_custom_op_symbolic('customop::reducel2', reducel2, 4)

inputs = get_dummy_input_for_rwkv_causal_llm(1, 1, model.device, model.args)
input_names = ['in'] + [f'state{j}_in' for j in range(3*model.layer_begin, 3*model.layer_end)]
output_names = ['out'] + [f'state{j}_out' for j in range(3*model.layer_begin, 3*model.layer_end)]
torch.onnx.export(model, tuple(inputs), args.output_name + ".onnx", input_names=input_names, output_names=output_names, opset_version=17)