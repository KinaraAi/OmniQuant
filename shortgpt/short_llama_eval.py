from tqdm import tqdm

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import os, sys
sys.path.insert(0, os.path.abspath('/auto/worka/aakash/llm/llama'))
from pathlib import Path
from llama import Llama
from short_llama import ShortLlama
from datautils import get_loaders
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	AutoConfig
)

seed = 10
MAX_SEQ_LEN = 2048  # authors use a context width of 1024
seqlen = MAX_SEQ_LEN

@torch.no_grad()
def evaluate(lm, tokenizer):
	results = {}
	lm.model = lm.model.to(lm.device)
	dataloader, testloader = get_loaders(
					"wikitext2",
					nsamples=128,
					seed=seed,
					model=tokenizer,
					seqlen=seqlen,
				)
	testenc = testloader.input_ids
	nsamples = min(300, testenc.numel() // seqlen)

	use_cache = lm.model.config.use_cache
	lm.model.eval()
	nlls = []
	for i in tqdm(range(nsamples)):
		batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(lm.device)
		outputs = lm.model.model(batch)
		hidden_states = outputs[0]
		logits = lm.model.lm_head(hidden_states)
		shift_logits = logits[:, :-1, :]
		shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
					:, 1:
				].to(lm.model.lm_head.weight.device)
		loss_fct = torch.nn.CrossEntropyLoss()
		loss = loss_fct(
					shift_logits.view(-1, shift_logits.size(-1)),
					shift_labels.view(-1),
				)
		neg_log_likelihood = loss.float() * seqlen
		nlls.append(neg_log_likelihood)

		ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
		print(f'wikitext2 : {ppl.item()}')
		lm.model.config.use_cache = use_cache
		results['wikitext2'] = ppl.item()
	return results

data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test[:100]')
dataloader = DataLoader(
	data,
	batch_size=1,
	shuffle=True,
	generator=torch.Generator(device="cuda")
)

print("Loading model")
llama = Llama.build(
	ckpt_dir="/auto/worka/aakash/llm/llama/llama-2-7b",
	tokenizer_path="/auto/worka/aakash/llm/llama/tokenizer.model",
	max_seq_len=MAX_SEQ_LEN,
	max_batch_size=1,
)

short_llama = ShortLlama(llama=llama, n_prune_layers=6)

print("Llama Model Layers")
print(short_llama.llama.model.layers)


text_gen = short_llama.llama.text_completion(
	prompts=["I am an avid fan of "],
	max_gen_len=30
)

print(text_gen)


for batch in tqdm(dataloader):
	prompts = batch['text']

	prompt_tokens = [short_llama.llama.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
	max_prompt_len = max(len(t) for t in prompt_tokens)

	# authors use a sliding window of size 1024 with a shift of 256
	for start in range(0, max_prompt_len, 256):

		inputs = [p[start:start+MAX_SEQ_LEN] for p in prompt_tokens if len(p) > start]

		short_llama.eval_importance(
			prompt_tokens=inputs,
			max_gen_len=0,
            angular=True
		)
  
print("Importances")
print(short_llama.importances)
print("*"*50)

exit()
def build_model_and_tokenizer(model_name):
	kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = None
	return model, tokenizer

model, tokenizer = build_model_and_tokenizer("/auto/worka/aakash/llm/llama/llama_output_dir")
testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
testenc = testenc.input_ids
ppl = short_llama.perplexity(testenc)
print(f"Perplexity : {ppl}")

print(100*"*")
print("Post Pruning")
short_llama.remove_layers(angular = True)

print(short_llama.llama.model.layers)

text_gen = short_llama.llama.text_completion(
	prompts=["I am an avid fan of "],
	max_gen_len=30
)

print(text_gen)
ppl = short_llama.perplexity(testenc)
print(f"Perplexity : {ppl}")

