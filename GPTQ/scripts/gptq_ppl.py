from transformers import AutoTokenizer , AutoModelForCausalLM
import torch, json, os
import functools, copy
from datetime import datetime
import torch.nn as nn
from datautils import get_loaders
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import argparse, random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--calib_dataset",type=str,default="wikitext2", 
					choices=["wikitext2", "ptb", "c4", "mix","pile"], help="Where to extract calibration data from.")

args = parser.parse_args()

# Load the Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",torch_dtype = "auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
print("Loaded model into device:", model.device)



def calibrate_model(model, dataset):
	# model.to(device)
	print("Model in device: ", model.device)
	seqlen = 2048
	cache_testloader = os.path.join(os.getcwd(), f'cache/testloader_qwen2_7b_{dataset}_all.cache')
	if os.path.exists(cache_testloader): 
		testloader = torch.load(cache_testloader)
		print(f"load calibration from {cache_testloader}")
	else:
		dataloader, testloader = get_loaders(
					dataset,
					seed=2,
					model=args.model,
					seqlen=seqlen,
				)
		torch.save(testloader, cache_testloader)
	if "c4" in dataset:
		testenc = testloader
	else:
		testenc = testloader.input_ids
	
	# nsamples = min(10,testenc.numel() // seqlen)
	nsamples = testenc.numel() // seqlen
	use_cache = model.config.use_cache
	model.config.use_cache = False
	model.eval()
	nlls = []
	for i in tqdm(range(nsamples)):
		with torch.no_grad():
			batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
			logits = model(batch)['logits'].to('cpu')
			shift_logits = logits[:, :-1, :]
			shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
							:, 1:]
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(
							shift_logits.view(-1, shift_logits.size(-1)),
							shift_labels.view(-1),
						)
			neg_log_likelihood = loss.float() * seqlen
			nlls.append(neg_log_likelihood)
			del batch, logits, shift_logits, shift_labels
			torch.cuda.empty_cache()

	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
	print(f'{dataset} : {ppl.item()}')
	model.config.use_cache = use_cache



calibrate_model(model, args.calib_dataset)





