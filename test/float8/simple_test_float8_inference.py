import torch
import torch.nn as nn
import torchao

from torchao.float8.inference import (
    ActivationCasting,
    Float8InferenceLinear,
    QuantConfig,
    quantize_to_float8,
    FbgemmFloat8InferenceLinear,
)

import fbgemm_gpu.experimental.gen_ai

device = torch.device('cuda')
M = 16
N = 4096
K = 14336
torch.set_default_device(device)

regular_linear = nn.Linear(K, N, bias=False, dtype=torch.bfloat16)
regular_linear.reset_parameters()
bf16_weight = torch.tensor(regular_linear.weight)
input_act = torch.randn((M, K), dtype=torch.bfloat16)

with torch.no_grad():
    out_bf16 = regular_linear(input_act)

print(f"{out_bf16=}")

# torchao.float8
quant_config = QuantConfig(ActivationCasting.DYNAMIC)
fp8_linear = quantize_to_float8(regular_linear, quant_config) # returns Float8InferenceLinear
print(fp8_linear)
print(f"{fp8_linear.weight._scale=}")
with torch.no_grad():
    out_fp8 = fp8_linear(input_act)

print(f"{out_fp8=}")
print(f"l2norm={(out_bf16-out_fp8).norm(p=2)}")

# fbgemm.gen_ai
aq, ascale = torch.ops.fbgemm.quantize_fp8_per_row(input_act)
wq, wscale = torch.ops.fbgemm.quantize_fp8_per_row(bf16_weight)
print(f"{wscale=}")

with torch.no_grad(): 
    out_fbgemm = torch.ops.fbgemm.f8f8bf16_rowwise(aq, wq, ascale, wscale, use_fast_accum=True)

print(f"{out_fbgemm=}")
print(f"l2norm={(out_bf16-out_fbgemm).norm(p=2)}")

# NEW Float8FbgemmInferenceLinear dynamic activation scale
quant_config = QuantConfig(activation_casting=ActivationCasting.DYNAMIC, 
                           dynamic_activation_quantization_ub=torch.tensor(1200.0),
               )
fbgemm_fp8_linear = FbgemmFloat8InferenceLinear(quant_config, K, N, bias=False)
fbgemm_fp8_linear.load_parameters(quantized_weight=wq, weight_scale=wscale)

with torch.no_grad(): 
    out_fbgemm_linear = fbgemm_fp8_linear(input_act)

print(f"{out_fbgemm_linear=}")
print(f"l2norm={(out_bf16-out_fbgemm_linear).norm(p=2)}")

# NEW Float8FbgemmInferenceLinear static activation scale
quant_config = QuantConfig(activation_casting=ActivationCasting.STATIC, 
                        # Even though the kernel takes scalar input scale here, 
                        # the results will have large error compared to using a column
                        # vector
                        #  static_quantization_scale=torch.tensor(1.0),
                           static_quantization_scale=torch.full((M, 1), 1.0),
               )
fbgemm_fp8_linear = FbgemmFloat8InferenceLinear(quant_config, K, N, bias=False)
fbgemm_fp8_linear.load_parameters(quantized_weight=wq, weight_scale=wscale)

with torch.no_grad(): 
    out_fbgemm_linear = fbgemm_fp8_linear(input_act)

print(f"{out_fbgemm_linear=}")
print(f"l2norm={(out_bf16-out_fbgemm_linear).norm(p=2)}")


