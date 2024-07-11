from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse, torch, functools

class BaseHook(object):
	def __init__(self):
		self.sign = {}
		self.n_bits = {}
	
	def __call__(self, name):
		self.sign[name] = True
		self.n_bits[name] = 8

class MinMaxHook(BaseHook):
	def __init__(self):
		super().__init__()
		self.min_vals = {}
		self.max_vals = {}

	def __call__(self, module, input, output, name):
		super().__call__(name)
		if True:
			breakpoint()
			# Extract the minimum and maximum values from the output tensor
			if name not in self.min_vals:
				self.min_vals[name] = output.min().item()
				self.max_vals[name] = output.max().item()
			else:
				self.min_vals[name] = min(self.min_vals[name], output.min().item())
				self.max_vals[name] = max(self.max_vals[name], output.max().item())
	
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
args = parser.parse_args()

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
	args.model,
	torch_dtype="auto",
	device_map="auto",
 	trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

prompt = "Give me a short introduction to large language model."
messages = [
	{"role": "system", "content": "You are a helpful assistant."},
	{"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
	messages,
	tokenize=False,
	add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

def add_stat_hooks(model):
	hooks = []
	hook = MinMaxHook()
	for name, module in model.named_modules():
		if "lm_head" in name:
			hooks.append(
				module.register_forward_hook(
					functools.partial(hook, name = name)
				)
			)

	return model, hooks, hook

# model, removable_handles, stat_hook = add_stat_hooks(model)

generated_ids = model.generate(
	model_inputs.input_ids,
	max_new_tokens=50
)
generated_ids = [
	output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
