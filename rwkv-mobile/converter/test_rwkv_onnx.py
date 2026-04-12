from pathlib import Path
import argparse, types, os
import numpy as np
from transformers import AutoTokenizer
import time
import onnxruntime

parser = argparse.ArgumentParser(description='Test coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV mlpackage file')
parser_args = parser.parse_args()

model = onnxruntime.InferenceSession(str(parser_args.model))

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)

num_layers = len(model.get_inputs()) // 3
hidden_size = model.get_inputs()[1].shape[-1]
num_heads = model.get_inputs()[2].shape[-3]
head_size = model.get_inputs()[2].shape[-1]
dtype = model.get_inputs()[1].type
np_dtype = np.float32 if dtype == 'tensor(float)' else np.float16

print(f'num_layers: {num_layers}, hidden_size: {hidden_size}, num_heads: {num_heads}, head_size: {head_size}, dtype: {dtype}')
prompt = "The Eiffel Tower is in the city of"
print(prompt, end='', flush=True)

inputs = {'in': np.array([[0]], dtype=np.int64)}
for i in range(num_layers):
    inputs[f'state{3*i}_in'] = np.zeros((1, 1, hidden_size), dtype=np_dtype)
    inputs[f'state{3*i+1}_in'] = np.zeros((1, num_heads, head_size, head_size), dtype=np_dtype)
    inputs[f'state{3*i+2}_in'] = np.zeros((1, 1, hidden_size), dtype=np_dtype)

for id in tokenizer.encode(prompt):
    inputs['in'][0][0] = id
    outputs = model.run(None, inputs)
    for i in range(num_layers):
        inputs[f'state{3*i}_in'] = outputs[3*i+1]
        inputs[f'state{3*i+1}_in'] = outputs[3*i+2]
        inputs[f'state{3*i+2}_in'] = outputs[3*i+3]

# calculate the durations
durations = []
for i in range(128):
    token_id = np.argmax(outputs[0][0])
    inputs['in'][0][0] = token_id
    print(tokenizer.decode([token_id]), end='', flush=True)
    for i in range(num_layers):
        inputs[f'state{3*i}_in'] = outputs[3*i+1]
        inputs[f'state{3*i+1}_in'] = outputs[3*i+2]
        inputs[f'state{3*i+2}_in'] = outputs[3*i+3]
    start_time = time.time()
    outputs = model.run(None, inputs)
    durations.append((time.time() - start_time) * 1000)  # convert to milliseconds

avg_duration = sum(durations) / len(durations)
print(f"\n\nAverage prediction time: {avg_duration:.2f} ms")
print(f"Tokens per second: {1000 / avg_duration:.2f}")

# outputs = model.run(None, inputs)
# print(outputs)