from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch, os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--output_folder", default="gptq_model", type=str, help="Path to save the merged model")
parser.add_argument("--calib_dataset",type=str,default="wikitext2", 
					choices=['wikitext2','c4','c4-new','ptb','ptb-new'], help="Where to extract calibration data from.")
parser.add_argument("--bits",type=int,default=4, 
					choices=[2,4,6,8], help="Number of bits to quantize the model to")
parser.add_argument("--group_size",type=int,default=128, 
					choices=[-1,16,32,64,128], help="group size for the group quantization")

args = parser.parse_args()

examples =  [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
]

def get_wikitext2():
    
    return []
quantization_config = GPTQConfig(
     bits=args.bits, # Supported precisions are [2, 4, 6, 8]
     group_size=args.group_size, # The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
     dataset = args.calib_dataset, # dataset= args.calib_dataset, # Supported default datasets ['wikitext2','c4','c4-new','ptb','ptb-new']
     desc_act=False, # Process rows based on decreasing activation. Places most of the quantization error on less significant weights
) 

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
quant_model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                quantization_config=quantization_config, 
                torch_dtype="auto",
                low_cpu_mem_usage=False,
                trust_remote_code=True)

print("Loaded Model")
# quant_model will have qweight and qzeros attributes in torch.int32 dtype.
output_path = os.path.join("/auto/regrt/sw/dgundimeda/GPTQ/output_models", args.output_folder)
tokenizer.save_pretrained(output_path)
quant_model.save_pretrained(output_path)
from transformers import pipeline

generator = pipeline('text-generation', model=quant_model, tokenizer=tokenizer)
result = generator("I have a dream", do_sample=True, max_length=50)[0]['generated_text']
print(result)

