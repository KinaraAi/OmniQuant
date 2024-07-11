import torch.nn as nn
import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	AutoConfig
)
import auto_gptq
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP
import copy
def convert_to_full_precision_old(model):
	for name, module in model.named_modules():
		if isinstance(module, auto_gptq.nn_modules.qlinear.qlinear_cuda_old.QuantLinear):
			# Access scale and zero_point (replace with your method)
			print(name)
			zeros = torch.bitwise_right_shift(torch.unsqueeze(module.qzeros, 2).cpu().expand(-1, -1, 32 // module.bits),module.wf.unsqueeze(0),).to(torch.int16 if module.bits == 8 else torch.int8)

			zeros = zeros + 1
			zeros = torch.bitwise_and(
				zeros, (2**module.bits) - 1
			)  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

			zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

			scales = module.scales
			scales = scales.reshape(-1, 1, scales.shape[-1])

			weight = torch.bitwise_right_shift(
					torch.unsqueeze(module.qweight, 1).cpu().expand(-1, 32 // module.bits, -1),
					module.wf.unsqueeze(-1),
				).to(torch.int16 if module.bits == 8 else torch.int8)
			weight = torch.bitwise_and(weight, (2**module.bits) - 1)
			weight = weight.reshape(-1, module.group_size, weight.shape[2])
			
			
			# Create full-precision layer
			full_precision_layer = nn.Linear(module.infeatures, module.outfeatures, bias = False)

			# Dequantize weights
			weight = scales.cpu() * (weight - zeros)
			weight_fp = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
			full_precision_layer.weight = torch.nn.Parameter(weight_fp.cuda())

			# (Optional) Update bias if applicable (similar dequantization for bias)

			# Replace quantized layer
			setattr(model, name, full_precision_layer)
			if name == "model.layers.31.mlp.up_proj":
				break
	return model

def dequntize_gptq_weights(module):
	if True:
		if isinstance(module, auto_gptq.nn_modules.qlinear.qlinear_cuda_old.QuantLinear):
			# Access scale and zero_point (replace with your method)
			zeros = torch.bitwise_right_shift(torch.unsqueeze(module.qzeros, 2).cpu().expand(-1, -1, 32 // module.bits),module.wf.unsqueeze(0),).to(torch.int16 if module.bits == 8 else torch.int8)

			zeros = zeros + 1
			zeros = torch.bitwise_and(
				zeros, (2**module.bits) - 1
			)  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

			zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

			scales = module.scales
			scales = scales.reshape(-1, 1, scales.shape[-1])

			weight = torch.bitwise_right_shift(
					torch.unsqueeze(module.qweight, 1).cpu().expand(-1, 32 // module.bits, -1),
					module.wf.unsqueeze(-1),
				).to(torch.int16 if module.bits == 8 else torch.int8)
			weight = torch.bitwise_and(weight, (2**module.bits) - 1)
			weight = weight.reshape(-1, module.group_size, weight.shape[2])
			
			
			# Create full-precision layer
			full_precision_layer = nn.Linear(module.infeatures, module.outfeatures, bias = False)

			# Dequantize weights
			weight = scales.cpu() * (weight - zeros)
			weight_fp = torch.transpose(weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]), 0,1)
			return torch.nn.Parameter(weight_fp.contiguous().cuda())
	print(50*"*")
	print("Error !!! Not a Quant Linear Layer ", module)
	print(50*"*")
	return None



def create_linear_module(module, bias_flag = False):
	new_module = nn.Linear(module.infeatures, module.outfeatures, bias=bias_flag)
	new_module.weight = dequntize_gptq_weights(module)
	if bias_flag:
		new_module.bias = torch.nn.Parameter(module.bias)
	return new_module


def sample_inference(model, tokenizer):
	eval_prompt = 'Simply put, the theory of relativity states that'
 #"The name of the capital of India is "
 #'My name is Julien and I like to'
 #'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
 #'Explain the plot of Cinderella in a sentence.'
	#'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'

	model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

	model.eval()
	with torch.no_grad():
		# print(tokenizer.decode(model.generate(**model_input, top_k=50, top_p=0.95, do_sample = True)[0], 
		# 		skip_special_tokens=True, do_sample=True,
		# 		num_return_sequences=3,
		# 		eos_token_id=tokenizer.eos_token_id,
		# 		max_length=100))
		print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True, do_sample=True,
	top_k=10,
	top_p = 0.95,
	num_return_sequences=1,
	eos_token_id=tokenizer.eos_token_id,
	max_length=100))
  
def dequantize_model(model):
	for name, m in model.model.named_modules():
		if isinstance(m, LlamaMLP) or isinstance(m, Qwen2MLP):
			m.gate_proj = create_linear_module(m.gate_proj)
			m.up_proj = create_linear_module(m.up_proj)
			m.down_proj = create_linear_module(m.down_proj)
			
		   
		elif isinstance(m, LlamaAttention) or isinstance(m, Qwen2Attention):
			m.q_proj = create_linear_module(m.q_proj, True)
			m.k_proj = create_linear_module(
				m.k_proj, True)
			m.v_proj = create_linear_module(
				m.v_proj, True)
			m.o_proj = create_linear_module(m.o_proj)
	return model

def convert_to_full_precision(model):
	for module in model.modules():
		breakpoint()
		if isinstance(module, auto_gptq.nn_modules.qlinear.qlinear_cuda_old.QuantLinear):
			# Access scale and zero_point (replace with your method)
			zeros = torch.bitwise_right_shift(torch.unsqueeze(module.qzeros, 2).cpu().expand(-1, -1, 32 // module.bits),module.wf.unsqueeze(0),).to(torch.int16 if module.bits == 8 else torch.int8)

			zeros = zeros + 1
			zeros = torch.bitwise_and(
				zeros, (2**module.bits) - 1
			)  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

			zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

			scales = module.scales
			scales = scales.reshape(-1, 1, scales.shape[-1])

			weight = torch.bitwise_right_shift(
					torch.unsqueeze(module.qweight, 1).cpu().expand(-1, 32 // module.bits, -1),
					module.wf.unsqueeze(-1),
				).to(torch.int16 if module.bits == 8 else torch.int8)
			weight = torch.bitwise_and(weight, (2**module.bits) - 1)
			weight = weight.reshape(-1, module.group_size, weight.shape[2])
			
			
			# Create full-precision layer
			full_precision_layer = nn.Linear(module.infeatures, module.outfeatures, bias = False)

			# Dequantize weights
			weight = scales.cpu() * (weight - zeros)
			weight_fp = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
			full_precision_layer.weight = torch.nn.Parameter(weight_fp.cuda())

			# (Optional) Update bias if applicable (similar dequantization for bias)

			# Replace quantized layer
			# setattr(model, name, full_precision_layer)
	return model

def build_model_and_tokenizer(model_name):
	kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
	return model, tokenizer

model_name = "/auto/regrt/sw/dgundimeda/GPTQ/output_models/qwen_1_5_base_int4_g128_wikitext2"
model, tokenizer = build_model_and_tokenizer(model_name)
# model_name = "/auto/worka/aakash/llm/llama/llama_output_dir/"
# original_model, tokenizer = build_model_and_tokenizer(model_name)


full_precision_model = dequantize_model(model)
save_path = "/auto/regrt/sw/dgundimeda/GPTQ/float_gptq_models/qwen_1_5_base_int4_g128_wikitext2_gptq_float_llamafied"
full_precision_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
sample_inference(full_precision_model, tokenizer)