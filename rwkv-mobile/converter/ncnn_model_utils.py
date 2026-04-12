import torch
import torch.nn.functional as F

def check_rwkv_info(state_dict):
    n_layer = 0
    version = 5
    n_head = 0
    head_size = 64
    for k in state_dict.keys():
        layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
        n_layer = max(n_layer, layer_id + 1)
        if 'ln_x' in k:
            version = max(5, version)
        if 'gate.weight' in k:
            version = max(5.1, version)
        if int(version) == 5 and 'att.time_decay' in k:
            n_head = state_dict[k].shape[0]
            if len(state_dict[k].shape) > 1:
                if state_dict[k].shape[1] > 1:
                    version = max(5.2, version)
        if 'time_maa' in k:
            version = max(6, version)
        if 'r_k' in k:
            version = max(7, version)
            n_head, _ = state_dict[k].shape
        if int(version) == 6 and 'time_faaaa' in k:
            n_head = state_dict[k].shape[0]
        if 'emb' in k:
            vocab_size, _ = state_dict[k].shape
    return version, n_layer, n_head, head_size, vocab_size

def write_weightdata(fp, dtype, tensor, with_flag=False):
    if dtype == torch.float32:
        if with_flag:
            fp.write(b'\x00\x00\x00\x00')
        fp.write(tensor.to(torch.float32).numpy().tobytes())
    elif dtype == torch.float16:
        if with_flag:
            fp.write(b'\x47\x6B\x30\x01')
        fp.write(tensor.to(torch.float16).numpy().tobytes())
        # align to 32bit
        if len(tensor.flatten()) % 2 != 0:
            fp.write(b'\x00\x00')
    else:
        assert 0, f'unsupported dtype {dtype}'

def build_info(param_lines, fp, version, n_layer, n_head, head_size, vocab_size, layer_count, blob_count):
    line = f"MemoryData model_info 0 1 model_info 0=5 21=1\n"
    param_lines.append(line)
    tensor = torch.tensor([version, n_layer, n_head, head_size, vocab_size], dtype=torch.float32)
    write_weightdata(fp, torch.float32, tensor)
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

reshape_count = 0
def build_reshape(param_lines, input, output, shape, layer_count, blob_count):
    global reshape_count
    line = f'Reshape reshape_{reshape_count} 1 1 {input} {output}'
    if len(shape) == 1:
        line += f' 0={shape[0]}'
    elif len(shape) == 2:
        line += f' 0={shape[1]} 1={shape[0]}'
    elif len(shape) == 3:
        line += f' 0={shape[2]} 1={shape[1]} 2={shape[0]}'
    elif len(shape) == 4:
        line += f' 0={shape[3]} 1={shape[2]} 2={shape[1]} 11={shape[0]}'
    else:
        assert 0, f'unsupported weight shape {shape}'
    line += '\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 1
    reshape_count += 1
    return layer_count, blob_count

def build_inp_emb(param_lines, fp, w, n_head, head_size, vocab_size, n_layers, layer_count, blob_count, use_fp32=False):
    param_lines.append('Input input_0 0 1 token 0=1 1=1 2=1\n')
    for i in range(n_layers):
        param_lines.append(f'Input input_{3 * i + 1} 0 1 state_{3 * i}_in 0={n_head * head_size}\n')
        param_lines.append(f'Input input_{3 * i + 2} 0 1 state_{3 * i + 1}_in 0={head_size} 1={head_size} 2={n_head}\n')
        param_lines.append(f'Input input_{3 * i + 3} 0 1 state_{3 * i + 2}_in 0={n_head * head_size}\n')
        layer_count += 3
        blob_count += 3
    param_lines.append(f'Embed embedding 1 1 token emb 0={n_head * head_size} 1={vocab_size} 3={n_head * head_size * vocab_size}\n')
    write_weightdata(fp, torch.float32 if use_fp32 else torch.float16, F.layer_norm(w['emb.weight'], w['emb.weight'].size()[-1:], weight=w['blocks.0.ln0.weight'].flatten(), bias=w['blocks.0.ln0.bias'].flatten()).half(), with_flag=True)
    layer_count += 2
    blob_count += 2
    return layer_count, blob_count

layernorm_count = 0
def build_layernorm(param_lines, fp, input, output, weight_gamma, weight_beta, layer_count, blob_count):
    assert len(weight_gamma.shape) == 1
    assert len(weight_beta.shape) == 1
    assert weight_gamma.shape[0] == weight_beta.shape[0]
    global layernorm_count
    param_lines.append(f'LayerNorm layernorm_{layernorm_count} 1 1 {input} {output} 0={weight_gamma.shape[0]} 1=0.00001 2=1\n')
    write_weightdata(fp, torch.float32, weight_gamma)
    write_weightdata(fp, torch.float32, weight_beta)
    layer_count += 1
    blob_count += 1
    layernorm_count += 1
    return layer_count, blob_count

sub_count = 0
def build_sub(param_lines, input1, input2, output, layer_count, blob_count):
    global sub_count
    param_lines.append(f'BinaryOp sub_{sub_count} 2 1 {input1} {input2} {output} 0=1\n')
    layer_count += 1
    blob_count += 1
    sub_count += 1
    return layer_count, blob_count

add_count = 0
def build_add(param_lines, input1, input2, output, layer_count, blob_count, scalar_B=False):
    global add_count
    if scalar_B == False:
        param_lines.append(f'BinaryOp add_{add_count} 2 1 {input1} {input2} {output} 0=0\n')
    else:
        param_lines.append(f'BinaryOp add_{add_count} 1 1 {input1} {output} 0=0 1=1 2={input2}\n')
    layer_count += 1
    blob_count += 1
    add_count += 1
    return layer_count, blob_count

minus_count = 0
def build_minus(param_lines, input1, input2, output, layer_count, blob_count):
    global minus_count
    param_lines.append(f'BinaryOp minus_{minus_count} 2 1 {input1} {input2} {output} 0=1\n')
    layer_count += 1
    blob_count += 1
    minus_count += 1
    return layer_count, blob_count

mul_count = 0
def build_mul(param_lines, input1, input2, output, layer_count, blob_count, scalar_B=False):
    global mul_count
    if scalar_B == False:
        param_lines.append(f'BinaryOp mul_{mul_count} 2 1 {input1} {input2} {output} 0=2\n')
    else:
        param_lines.append(f'BinaryOp mul_{mul_count} 1 1 {input1} {output} 0=2 1=1 2={input2}\n')
    layer_count += 1
    blob_count += 1
    mul_count += 1
    return layer_count, blob_count

def build_split(param_lines, input, output_list, layer_count, blob_count):
    line = f"Split split_{input} 1 {len(output_list)} {input}"
    for output in output_list:
        line += f' {output}'
    line += '\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += len(output_list)
    return layer_count, blob_count

def build_data(param_lines, fp, weight, name, dtype, layer_count, blob_count):
    line = f"MemoryData data_{name} 0 1 {name}"
    if len(weight.size()) == 1:
        line += f' 0={weight.size()[0]}'
    elif len(weight.size()) == 2:
        line += f' 0={weight.size()[1]} 1={weight.size()[0]}'
    elif len(weight.size()) == 3:
        line += f' 0={weight.size()[2]} 1={weight.size()[1]} 2={weight.size()[0]}'
    elif len(weight.size()) == 4:
        line += f' 0={weight.size()[3]} 1={weight.size()[2]} 2={weight.size()[1]} 11={weight.size()[0]}'
    else:
        assert 0, f'unsupported weight shape {weight.size()}'

    # auto dtype
    line += ' 21=0\n'
    param_lines.append(line)
    write_weightdata(fp, dtype, weight, with_flag=True)
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

sigmoid_count = 0
def build_sigmoid(param_lines, input, output, layer_count, blob_count):
    global sigmoid_count
    param_lines.append(f'Sigmoid sigmoid_{sigmoid_count} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    sigmoid_count += 1
    return layer_count, blob_count

relu_count = 0
def build_relu(param_lines, input, output, layer_count, blob_count):
    global relu_count
    param_lines.append(f'ReLU relu_{relu_count} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    relu_count += 1
    return layer_count, blob_count

def build_square(param_lines, input, output, layer_count, blob_count):
    param_lines.append(f'UnaryOp square_{input} 1 1 {input} {output} 0=4\n')
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_tanh(param_lines, input, output, layer_count, blob_count):
    param_lines.append(f'TanH tanh_{input} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_linear(param_lines, fp, input, output, weight, dtype, layer_count, blob_count):
    param_lines.append(f'Gemm gemm_{output} 1 1 {input} {output} 4=0 5=1 6=0 7=0 8={weight.shape[0]} 9={weight.shape[1]} 10=-1\n')
    write_weightdata(fp, dtype, weight.t().flatten(), with_flag=True)
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_matmul(param_lines, input1, input2, output, layer_count, blob_count):
    param_lines.append(f'MatMul matmul_{output} 2 1 {input1} {input2} {output} 0=0\n')
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_squeeze(param_lines, input, output, dim, layer_count, blob_count):
    line = f'Squeeze squeeze_{output} 1 1 {input} {output}'
    for d in dim:
        if d == 3:
            line += ' 11=1'
        else:
            line += f' {d}=1'
    line += '\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_slice_wkvrg(param_lines, input, outputs, layer_count, blob_count):
    line = f'Slice slice_{input} 1 5 {input}'
    assert len(outputs) == 5
    for output in outputs:
        line += f' {output}'
    line += ' -23300=5,-233,-233,-233,-233,-233 1=0\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 5
    return layer_count, blob_count

def build_slice_rwkvag(param_lines, input, outputs, layer_count, blob_count):
    line = f'Slice slice_{input} 1 6 {input}'
    assert len(outputs) == 6
    for output in outputs:
        line += f' {output}'
    line += ' -23300=6,-233,-233,-233,-233,-233,-233 1=0\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 6
    return layer_count, blob_count

groupnorm_count = 0
def build_groupnorm(param_lines, fp, input, output, weight, bias, num_groups, layer_count, blob_count, eps='0.00001'):
    global groupnorm_count
    param_lines.append(f'GroupNorm groupnorm_{groupnorm_count} 1 1 {input} {output} 0={num_groups} 1={weight.flatten().shape[0]} 2={eps} 3=1\n')
    layer_count += 1
    blob_count += 1
    groupnorm_count += 1
    write_weightdata(fp, torch.float32, weight)
    write_weightdata(fp, torch.float32, bias)
    return layer_count, blob_count

l2norm_count = 0
def build_l2norm(param_lines, fp, input, output, layer_count, blob_count):
    global l2norm_count
    param_lines.append(f'Normalize l2norm_{l2norm_count} 1 1 {input} {output} 0=1 1=1 2=0.0000001 3=1 4=0 9=1\n')
    write_weightdata(fp, torch.float32, torch.tensor([1.0]))
    layer_count += 1
    blob_count += 1
    l2norm_count += 1
    return layer_count, blob_count

exp_count = 0
def build_exp(param_lines, input, output, layer_count, blob_count, scale="1.0"):
    global exp_count
    if scale != "1.0":
        param_lines.append(f'Exp exp_{exp_count} 1 1 {input} {output} 1={scale}\n')
    else:
        param_lines.append(f'Exp exp_{exp_count} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    exp_count += 1
    return layer_count, blob_count

sum_count = 0
def build_sum(param_lines, input, output, layer_count, blob_count):
    global sum_count
    line = f'Reduction sum_{sum_count} 1 1 {input} {output} 1=0 -23303=1,-1 4=1 5=1\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 1
    sum_count += 1
    return layer_count, blob_count

cast_count = 0
def build_cast(param_lines, input, output, intype, outtype, layer_count, blob_count):
    global cast_count
    if intype == torch.float32:
        intype = 1
    elif intype == torch.float16:
        intype = 2
    else:
        assert 0, f'unsupported intype {intype}'

    if outtype == torch.float32:
        outtype = 1
    elif outtype == torch.float16:
        outtype = 2
    else:
        assert 0, f'unsupported outtype {outtype}'

    param_lines.append(f'Cast cast_{cast_count} 1 1 {input} {output} 0={intype} 1={outtype}\n')
    layer_count += 1
    blob_count += 1
    cast_count += 1
    return layer_count, blob_count

def build_time_mixing_v6(param_lines, fp, w, input, output, layer_id, layer_count, blob_count, n_head, head_size, jellyfish_modified=False, use_fp32=False):
    prefix = f'att_{layer_id}_'
    weight_dtype = torch.float32 if use_fp32 else torch.float16
    layer_count, blob_count = build_split(param_lines, input, [prefix + 'x_last', prefix + 'x'], layer_count, blob_count)
    layer_count, blob_count = build_layernorm(param_lines, fp, prefix + 'x', prefix + 'xx', w[f'blocks.{layer_id}.ln1.weight'], w[f'blocks.{layer_id}.ln1.bias'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'xx', [prefix + 'xx_0', prefix + 'xx_1', prefix + 'xx_2', f'state_{3*layer_id}_out'], layer_count, blob_count)

    # sub_shifted
    layer_count, blob_count = build_sub(param_lines, f'state_{3*layer_id}_in', prefix + 'xx_0', prefix + 'sx', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'sx', [prefix + 'sx_0', prefix + 'sx_1'], layer_count, blob_count)

    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_maa_x'].flatten(), prefix + 'maa_x', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_0', prefix + 'maa_x', prefix + 'maa_xx', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xx_1', prefix + 'maa_xx', prefix + 'xxx', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'xxx', prefix + 'maa_x_lora', w[f'blocks.{layer_id}.att.time_maa_w1'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_tanh(param_lines, prefix + 'maa_x_lora', prefix + 'maa_x_lora_tanh', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'maa_x_lora_tanh', prefix + 'maa_x_lora_tanh_reshape', [5, 1, -1], layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_maa_w2'], prefix + 'maa_w2', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'maa_x_lora_tanh_reshape', prefix + 'maa_w2', prefix + 'maa_x_post_lora', layer_count, blob_count)

    w_maa = torch.cat([w[f'blocks.{layer_id}.att.time_maa_w'], w[f'blocks.{layer_id}.att.time_maa_k'], w[f'blocks.{layer_id}.att.time_maa_v'], w[f'blocks.{layer_id}.att.time_maa_r'], w[f'blocks.{layer_id}.att.time_maa_g']], dim=0)
    layer_count, blob_count = build_data(param_lines, fp, w_maa, prefix + 'maa', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'maa_x_post_lora', prefix + 'maa', prefix + 'maa_wkvrg_pre', layer_count, blob_count)
    layer_count, blob_count = build_squeeze(param_lines, prefix + 'maa_wkvrg_pre', prefix + 'maa_wkvrg_pre_squeezed', [1], layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_1', prefix + 'maa_wkvrg_pre_squeezed', prefix + 'maa_wkvrg_sx', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xx_2', prefix + 'maa_wkvrg_sx', prefix + 'maa_wkvrg', layer_count, blob_count)
    layer_count, blob_count = build_slice_wkvrg(param_lines, prefix + 'maa_wkvrg', [prefix + 'mw', prefix + 'mk', prefix + 'mv', prefix + 'mr', prefix + 'mg'], layer_count, blob_count)

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mw', prefix + 'mw_lora', w[f'blocks.{layer_id}.att.time_decay_w1'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_tanh(param_lines, prefix + 'mw_lora', prefix + 'mw_lora_tanh', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mw_lora_tanh', prefix + 'mw_lora_tanh_linear', w[f'blocks.{layer_id}.att.time_decay_w2'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_decay'].flatten(), prefix + 'td', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'mw_lora_tanh_linear', prefix + 'td', prefix + 'time_decay_pre', layer_count, blob_count)
    layer_count, blob_count = build_exp(param_lines, prefix + 'time_decay_pre', prefix + 'time_decay_exp0', layer_count, blob_count)
    layer_count, blob_count = build_exp(param_lines, prefix + 'time_decay_exp0', prefix + 'time_decay', layer_count, blob_count, scale="-1.0")

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mk', prefix + 'key', (w[f'blocks.{layer_id}.att.key.weight'] / 2), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mv', prefix + 'value', (w[f'blocks.{layer_id}.att.value.weight'] / 4), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mr', prefix + 'receptance', w[f'blocks.{layer_id}.att.receptance.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mg', prefix + 'gate', w[f'blocks.{layer_id}.att.gate.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'gate', [prefix + 'gate_0', prefix + 'gate_1'], layer_count, blob_count)
    layer_count, blob_count = build_sigmoid(param_lines, prefix + 'gate_0', prefix + 'gate_sigmoid', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'gate_1', prefix + 'gate_sigmoid', prefix + 'gate_silu', layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_faaaa'].unsqueeze(-1), prefix + 'time_first', weight_dtype, layer_count, blob_count)

    # non-customlayer implementation
    layer_count, blob_count = build_reshape(param_lines, prefix + 'key', prefix + 'key_reshape', [n_head, head_size, 1], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'value', prefix + 'value_reshape', [n_head, 1, head_size], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'receptance', prefix + 'receptance_reshape', [n_head, 1, head_size], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'time_decay', prefix + 'time_decay_reshape', [n_head, head_size, 1], layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'key_reshape', prefix + 'value_reshape', prefix + 'kv', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'kv', [prefix + 'kv_0', prefix + 'kv_1'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, f'state_{3*layer_id+1}_in', [prefix + 'wkv_state_0', prefix + 'wkv_state_1'], layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'kv_0', prefix + 'time_first', prefix + 'kv_time_first', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'kv_time_first', prefix + 'wkv_state_0', prefix + 'kv_tf_state', layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'receptance_reshape', prefix + 'kv_tf_state', prefix + 'wkv_out', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'wkv_state_1', prefix + 'time_decay_reshape', prefix + 'state_td', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'state_td', prefix + 'kv_1', f'state_{3*layer_id+1}_out', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'wkv_out', prefix + 'wkv_out_flatten', [n_head * head_size], layer_count, blob_count)

    layer_count, blob_count = build_groupnorm(param_lines, fp, prefix + 'wkv_out_flatten', prefix + 'x_gn', w[f'blocks.{layer_id}.att.ln_x.weight'].flatten(), w[f'blocks.{layer_id}.att.ln_x.bias'].flatten(), n_head, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'x_gn', prefix + 'gate_silu', prefix + 'x_gate', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'x_gate', prefix + 'x_out', w[f'blocks.{layer_id}.att.output.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'x_out', prefix + 'x_last', output, layer_count, blob_count)
    return layer_count, blob_count

def build_channel_mixing_v6(param_lines, fp, w, input, output, layer_id, layer_count, blob_count, use_fp32=False):
    prefix = f'ffn_{layer_id}_'
    weight_dtype = torch.float32 if use_fp32 else torch.float16
    layer_count, blob_count = build_split(param_lines, input, [prefix + 'x_last', prefix + 'x'], layer_count, blob_count)
    layer_count, blob_count = build_layernorm(param_lines, fp, prefix + 'x', prefix + 'xx', w[f'blocks.{layer_id}.ln2.weight'], w[f'blocks.{layer_id}.ln2.bias'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'xx', [prefix + 'xx_0', prefix + 'xx_1', prefix + 'xx_2', f'state_{3*layer_id+2}_out'], layer_count, blob_count)

    # sub_shifted
    layer_count, blob_count = build_sub(param_lines, f'state_{3*layer_id+2}_in', prefix + 'xx_0', prefix + 'sx', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'sx', [prefix + 'sx_0', prefix + 'sx_1'], layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.ffn.time_maa_k'].flatten(), prefix + 'maa_k', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.ffn.time_maa_r'].flatten(), prefix + 'maa_r', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_0', prefix + 'maa_k', prefix + 'xk', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_1', prefix + 'maa_r', prefix + 'xr', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xk', prefix + 'xx_1', prefix + 'xxk', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xr', prefix + 'xx_2', prefix + 'xxr', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'xxk', prefix + 'key', w[f'blocks.{layer_id}.ffn.key.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'xxr', prefix + 'receptance', w[f'blocks.{layer_id}.ffn.receptance.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_sigmoid(param_lines, prefix + 'receptance', prefix + 'receptance_sigmoid', layer_count, blob_count)
    layer_count, blob_count = build_relu(param_lines, prefix + 'key', prefix + 'key_relu', layer_count, blob_count)
    layer_count, blob_count = build_square(param_lines, prefix + 'key_relu', prefix + 'key_relu_square', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'key_relu_square', prefix + 'value', w[f'blocks.{layer_id}.ffn.value.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'value', prefix + 'receptance_sigmoid', prefix + 'rv', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'rv', prefix + 'x_last', output, layer_count, blob_count)
    return layer_count, blob_count

def build_time_mixing_v7(param_lines, fp, w, input, output, layer_id, layer_count, blob_count, n_head, head_size, jellyfish_modified=False, use_fp32=False):
    prefix = f'att_{layer_id}_'
    weight_dtype = torch.float32 if use_fp32 else torch.float16
    layer_count, blob_count = build_split(param_lines, input, [prefix + 'x_last', prefix + 'x'], layer_count, blob_count)
    layer_count, blob_count = build_layernorm(param_lines, fp, prefix + 'x', prefix + 'xx', w[f'blocks.{layer_id}.ln1.weight'], w[f'blocks.{layer_id}.ln1.bias'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'xx', [prefix + 'xx_0', prefix + 'xx_1', f'state_{3*layer_id}_out'], layer_count, blob_count)

    # sub_shifted
    layer_count, blob_count = build_sub(param_lines, f'state_{3*layer_id}_in', prefix + 'xx_0', prefix + 'sx', layer_count, blob_count)
    w_maa = torch.cat([w[f'blocks.{layer_id}.att.x_r'].squeeze(0), w[f'blocks.{layer_id}.att.x_w'].squeeze(0), w[f'blocks.{layer_id}.att.x_k'].squeeze(0), w[f'blocks.{layer_id}.att.x_v'].squeeze(0), w[f'blocks.{layer_id}.att.x_a'].squeeze(0), w[f'blocks.{layer_id}.att.x_g'].squeeze(0)], dim=0)
    layer_count, blob_count = build_data(param_lines, fp, w_maa, prefix + 'maa', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx', prefix + 'maa', prefix + 'maa_x', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xx_1', prefix + 'maa_x', prefix + 'maa_rwkvag', layer_count, blob_count)
    layer_count, blob_count = build_slice_rwkvag(param_lines, prefix + 'maa_rwkvag', [prefix + 'mr', prefix + 'mw', prefix + 'mk', prefix + 'mv', prefix + 'ma', prefix + 'mg'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'mv', [prefix + 'mv_0', prefix + 'mv_1'], layer_count, blob_count)

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mk', prefix + 'key', (w[f'blocks.{layer_id}.att.key.weight']), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mv_0', prefix + 'value', (w[f'blocks.{layer_id}.att.value.weight']), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mr', prefix + 'receptance', w[f'blocks.{layer_id}.att.receptance.weight'], weight_dtype, layer_count, blob_count)

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'ma', prefix + 'ma_lora_0', w[f'blocks.{layer_id}.att.a1'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'ma_lora_0', prefix + 'ma_lora_1', w[f'blocks.{layer_id}.att.a2'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.a0'].flatten(), prefix + 'a0', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'ma_lora_1', prefix + 'a0', prefix + 'ma_lora_2', layer_count, blob_count)
    if jellyfish_modified:
        layer_count, blob_count = build_sigmoid(param_lines, prefix + 'ma_lora_2', prefix + 'a_tmp', layer_count, blob_count)
        layer_count, blob_count = build_mul(param_lines, prefix + 'a_tmp', "2.0", prefix + 'a', layer_count, blob_count, scalar_B=True)
    else:
        layer_count, blob_count = build_sigmoid(param_lines, prefix + 'ma_lora_2', prefix + 'a', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'a', [prefix + 'a_0', prefix + 'a_1'], layer_count, blob_count)

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mg', prefix + 'mg_lora', w[f'blocks.{layer_id}.att.g1'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_sigmoid(param_lines, prefix + 'mg_lora', prefix + 'mg_lora_sigmoid', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mg_lora_sigmoid', prefix + 'gate', w[f'blocks.{layer_id}.att.g2'].t(), weight_dtype, layer_count, blob_count)

    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.k_k'].flatten(), prefix + 'k_k', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.k_a'].flatten(), prefix + 'k_a', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.r_k'].flatten(), prefix + 'r_k', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'key', [prefix + 'key_0', prefix + 'key_1'], layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'key_0', prefix + 'k_k', prefix + 'key_k', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'key_k', prefix + 'key_k_reshape', [n_head, 1, head_size], layer_count, blob_count)
    layer_count, blob_count = build_l2norm(param_lines, fp, prefix + 'key_k_reshape', prefix + 'key_k_norm', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'key_k_norm', prefix + 'key_k_norm_reshape', [n_head*head_size], layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'a_0', '-1.0', prefix + 'a_0_minus', layer_count, blob_count, scalar_B=True)
    layer_count, blob_count = build_mul(param_lines, prefix + 'a_0_minus', prefix + 'k_a', prefix + 'a_minus1_mulka', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'a_minus1_mulka', '1.0', prefix + 'a_minus1_mulka_plus1', layer_count, blob_count, scalar_B=True)
    layer_count, blob_count = build_mul(param_lines, prefix + 'a_minus1_mulka_plus1', prefix + 'key_1', prefix + 'key_a', layer_count, blob_count)
    if layer_id == 0:
        layer_count, blob_count = build_split(param_lines, prefix + 'value', [prefix + 'value_final', 'v_first_0'], layer_count, blob_count)
    else:
        layer_count, blob_count = build_split(param_lines, prefix + 'value', [prefix + 'value_0', prefix + 'value_1'], layer_count, blob_count)
        layer_count, blob_count = build_split(param_lines, f'v_first_{layer_id-1}', [prefix + 'v_first', f'v_first_{layer_id}'], layer_count, blob_count)
        layer_count, blob_count = build_minus(param_lines, prefix + 'v_first', prefix + 'value_1', prefix + 'vfirst_minus_value', layer_count, blob_count)
        layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mv_1', prefix + 'mv_lora_0', w[f'blocks.{layer_id}.att.v1'].t(), weight_dtype, layer_count, blob_count)
        layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mv_lora_0', prefix + 'mv_lora_1', w[f'blocks.{layer_id}.att.v2'].t(), weight_dtype, layer_count, blob_count)
        layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.v0'].flatten(), prefix + 'v0', weight_dtype, layer_count, blob_count)
        layer_count, blob_count = build_add(param_lines, prefix + 'mv_lora_1', prefix + 'v0', prefix + 'mv_lora_2', layer_count, blob_count)
        layer_count, blob_count = build_sigmoid(param_lines, prefix + 'mv_lora_2', prefix + 'mv_lora_sigmoid', layer_count, blob_count)
        layer_count, blob_count = build_mul(param_lines, prefix + 'mv_lora_sigmoid', prefix + 'vfirst_minus_value', prefix + 'vfirst_minus_value_mul', layer_count, blob_count)
        layer_count, blob_count = build_add(param_lines, prefix + 'value_0', prefix + 'vfirst_minus_value_mul', prefix + 'value_final', layer_count, blob_count)

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mw', prefix + 'mw_lora', w[f'blocks.{layer_id}.att.w1'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_tanh(param_lines, prefix + 'mw_lora', prefix + 'mw_lora_tanh', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mw_lora_tanh', prefix + 'mw_lora_tanh_linear', w[f'blocks.{layer_id}.att.w2'].t(), weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.w0'].flatten(), prefix + 'td', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'mw_lora_tanh_linear', prefix + 'td', prefix + 'time_decay_pre', layer_count, blob_count)
    layer_count, blob_count = build_sigmoid(param_lines, prefix + 'time_decay_pre', prefix + 'time_decay_sigmoid', layer_count, blob_count)
    layer_count, blob_count = build_exp(param_lines, prefix + 'time_decay_sigmoid', prefix + 'time_decay', layer_count, blob_count, scale="-0.606531")
    if jellyfish_modified:
        layer_count, blob_count = build_split(param_lines, prefix + 'time_decay', [prefix + 'time_decay_0', prefix + 'time_decay_1'], layer_count, blob_count)
        layer_count, blob_count = build_reshape(param_lines, prefix + 'time_decay_0', prefix + 'time_decay_reshape', [n_head, 1, head_size], layer_count, blob_count)
    else:
        layer_count, blob_count = build_reshape(param_lines, prefix + 'time_decay', prefix + 'time_decay_reshape', [n_head, 1, head_size], layer_count, blob_count)

    # non-customlayer implementation
    layer_count, blob_count = build_split(param_lines, prefix + 'key_a', [prefix + 'key_a_0', prefix + 'key_a_1'], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'key_a_0', prefix + 'key_a_reshape', [n_head, 1, head_size], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'value_final', [prefix + 'value_final_0', prefix + 'value_final_1', prefix + 'value_final_2'], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'value_final_0', prefix + 'value_final_0_reshape', [n_head, head_size], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'value_final_1', prefix + 'value_final_1_reshape', [n_head, head_size, 1], layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'value_final_1_reshape', prefix + 'key_a_reshape', prefix + 'vk', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'key_k_norm_reshape', [prefix + 'kk_0', prefix + 'kk_1'], layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'kk_0', "-1.0", prefix + 'kk_0_neg', layer_count, blob_count, scalar_B=True)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'kk_0_neg', prefix + 'kk_0_neg_reshape', [n_head, head_size, 1], layer_count, blob_count)
    if jellyfish_modified:
        layer_count, blob_count = build_mul(param_lines, prefix + 'time_decay_1', prefix + 'a_1', prefix + 'a_extended', layer_count, blob_count)
        layer_count, blob_count = build_mul(param_lines, prefix + 'a_extended', prefix + 'kk_1', prefix + 'kk_a', layer_count, blob_count)
    else:
        layer_count, blob_count = build_mul(param_lines, prefix + 'kk_1', prefix + 'a_1', prefix + 'kk_a', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'kk_a', prefix + 'kk_a_reshape', [n_head, 1, head_size], layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'kk_0_neg_reshape', prefix + 'kk_a_reshape', prefix + 'ab', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, f'state_{3*layer_id+1}_in', [prefix + 'state_prev_0', prefix + 'state_prev_1'], layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'state_prev_0', prefix + 'time_decay_reshape', prefix + 'state_td', layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'state_prev_1', prefix + 'ab', prefix + 'sab', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'sab', prefix + 'vk', prefix + 'sab_vk', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'sab_vk', prefix + 'state_td', prefix + 'state_new', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'state_new', [prefix + 'state_new_0', f'state_{3*layer_id+1}_out'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'receptance', [prefix + 'receptance_0', prefix + 'receptance_1'], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'receptance_0', prefix + 'receptance_reshape', [n_head, head_size, 1], layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'state_new_0', prefix + 'receptance_reshape', prefix + 'wkv_out', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'wkv_out', prefix + 'wkv_out_flatten', [n_head * head_size], layer_count, blob_count)

    layer_count, blob_count = build_groupnorm(param_lines, fp, prefix + 'wkv_out_flatten', prefix + 'x_gn', w[f'blocks.{layer_id}.att.ln_x.weight'].flatten(), w[f'blocks.{layer_id}.att.ln_x.bias'].flatten(), n_head, layer_count, blob_count, eps='0.00064')
    layer_count, blob_count = build_mul(param_lines, prefix + 'receptance_1', prefix + 'key_a_1', prefix + 'receptance_key_a', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'receptance_key_a', prefix + 'r_k', prefix + 'rk', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'rk', prefix + 'rk_reshape', [n_head, head_size], layer_count, blob_count)
    layer_count, blob_count = build_sum(param_lines, prefix + 'rk_reshape', prefix + 'rk_sum', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'value_final_0_reshape', prefix + 'rk_sum', prefix + 'value_final_0_rk', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'value_final_0_rk', prefix + 'value_final_0_rk_reshape', [n_head*head_size], layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'value_final_0_rk_reshape', prefix + 'x_gn', prefix + 'x_gn_rkv', layer_count, blob_count)

    layer_count, blob_count = build_mul(param_lines, prefix + 'x_gn_rkv', prefix + 'gate', prefix + 'x_gate', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'x_gate', prefix + 'x_out', w[f'blocks.{layer_id}.att.output.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'x_out', prefix + 'x_last', output, layer_count, blob_count)
    return layer_count, blob_count

def build_channel_mixing_v7(param_lines, fp, w, input, output, layer_id, layer_count, blob_count, use_fp32=False):
    prefix = f'ffn_{layer_id}_'
    weight_dtype = torch.float32 if use_fp32 else torch.float16
    layer_count, blob_count = build_split(param_lines, input, [prefix + 'x_last', prefix + 'x'], layer_count, blob_count)
    layer_count, blob_count = build_layernorm(param_lines, fp, prefix + 'x', prefix + 'xx', w[f'blocks.{layer_id}.ln2.weight'], w[f'blocks.{layer_id}.ln2.bias'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'xx', [prefix + 'xx_0', prefix + 'xx_1', f'state_{3*layer_id+2}_out'], layer_count, blob_count)

    # sub_shifted
    layer_count, blob_count = build_sub(param_lines, f'state_{3*layer_id+2}_in', prefix + 'xx_0', prefix + 'sx', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'sx', [prefix + 'sx_0'], layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.ffn.x_k'].flatten(), prefix + 'x_k', weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_0', prefix + 'x_k', prefix + 'xk', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xk', prefix + 'xx_1', prefix + 'xxk', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'xxk', prefix + 'key', w[f'blocks.{layer_id}.ffn.key.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_relu(param_lines, prefix + 'key', prefix + 'key_relu', layer_count, blob_count)
    layer_count, blob_count = build_square(param_lines, prefix + 'key_relu', prefix + 'key_relu_square', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'key_relu_square', prefix + 'value', w[f'blocks.{layer_id}.ffn.value.weight'], weight_dtype, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'value', prefix + 'x_last', output, layer_count, blob_count)
    return layer_count, blob_count

def build_output_head(param_lines, fp, w, input, output, layer_count, blob_count, use_fp32=False):
    layer_count, blob_count = build_layernorm(param_lines, fp, input, 'norm_head', w['ln_out.weight'].flatten(), w['ln_out.bias'].flatten(), layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, 'norm_head', output, w['head.weight'], torch.float32 if use_fp32 else torch.float16, layer_count, blob_count)
    return layer_count, blob_count