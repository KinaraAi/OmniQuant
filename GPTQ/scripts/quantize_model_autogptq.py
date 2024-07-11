import time
import os
import logging

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
import torch
import torch.nn as nn
import argparse

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    logger = logging.getLogger(__name__)

    wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikilist = [' \n' if s == '' else s for s in wikidata['text'] ]

    text = ''.join(wikilist)
    logger.info("Tokenising wikitext2")
    trainenc = tokenizer(text, return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset

def get_c4(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset, load_from_disk
    traindata = load_from_disk("/auto/worka/dgundimeda/llama2_onnx/c4")

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        trainloader.append({'input_ids':inp,'attention_mask': attention_mask})

    return trainloader

def quantize(model_dir, output_dir, traindataset, bits, group_size, desc_act, damp, batch_size = 1, use_triton=False, trust_remote_code=False, dtype='float16'):
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp
    )

    if dtype == 'float16':
        torch_dtype  = torch.float16
    elif dtype == 'float32':
        torch_dtype  = torch.float32
    elif dtype == 'bfloat16':
        torch_dtype  = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    logger.info(f"Loading model from {model_dir} with trust_remote_code={trust_remote_code} and dtype={torch_dtype}")
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config=quantize_config, low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)

    logger.info(f"Starting quantization to {output_dir} with use_triton={use_triton}")
    start_time = time.time()
    model.quantize(traindataset, use_triton=use_triton, batch_size=batch_size)

    logger.info(f"Time to quantize model at {output_dir} with use_triton={use_triton}: {time.time() - start_time:.2f}")

    logger.info(f"Saving quantized model to {output_dir}")
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(path)
    logger.info("Done.")

if __name__ == "__main__":
    logger = logging.getLogger()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    '''
    python3 /auto/regrt/sw/dgundimeda/GPTQ/scripts/quantize_model_autogptq.py /auto/regrt/sw/dgundimeda/qwen_models/qwen2_7b_llamafied /auto/regrt/sw/dgundimeda/GPTQ/output_models \
        wikitext --bits 4 --group_size 128 --desc_act 0  --dtype bfloat16 --batch_size 4
    '''
    parser = argparse.ArgumentParser(description='quantise')
    parser.add_argument('pretrained_model_dir', type=str, help='Repo name')
    parser.add_argument('output_dir_base', type=str, help='Output base folder')
    parser.add_argument('dataset', type=str, help='Output base folder')
    parser.add_argument('--trust_remote_code', action="store_true", help='Trust remote code')
    parser.add_argument('--use_triton', action="store_true", help='Use Triton for quantization')
    parser.add_argument('--bits', type=int, nargs='+', default=[4], help='Quantize bit(s)')
    parser.add_argument('--group_size', type=int, nargs='+', default=[32, 128, 1024, -1], help='Quantize group size(s)')
    parser.add_argument('--damp', type=float, nargs='+', default=[0.01], help='Quantize damp_percent(s)')
    parser.add_argument('--desc_act', type=int, nargs='+', default=[0, 1], help='Quantize desc_act(s) - 1 = True, 0 = False')
    parser.add_argument('--dtype', type=str, choices=['float16', 'float32', 'bfloat16'], help='Quantize desc_act(s) - 1 = True, 0 = False')
    parser.add_argument('--seqlen', type=int, default=2048, help='Model sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Quantize batch size for processing dataset samples')
    parser.add_argument('--stop_file', type=str, help='Filename to look for to stop inference, specific to this instance')

    args = parser.parse_args()

    stop_file = args.stop_file or ""

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_dir, use_fast=True, trust_remote_code=args.trust_remote_code)

    if args.dataset == 'wikitext':
        traindataset = get_wikitext2(128, 0, args.seqlen, tokenizer)
    elif args.dataset == 'c4':
        traindataset = get_c4(128, 0, args.seqlen, tokenizer)
    else:
        logger.error(f"Unsupported dataset: {args.dataset}")
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    abort = False

    iterations=[]
    for bits in args.bits:
        for group_size in args.group_size:
            for desc_act in args.desc_act:
                for damp in args.damp:
                    desc_act = desc_act == 1 and True or False
                    iterations.append({"bits": bits, "group_size": group_size, "desc_act": desc_act, "damp": damp})

    num_iters = len(iterations)
    logger.info(f"Starting {num_iters} quantizations.")
    count=1
    for iter in iterations:
        if not os.path.isfile("/workspace/gptq-ppl-test/STOP") and not os.path.isfile(stop_file) and not abort:
            bits = iter['bits']
            group_size = iter['group_size']
            desc_act = iter['desc_act']
            damp = iter['damp']

            output_dir = args.output_dir_base
            try:
                os.makedirs(output_dir, exist_ok=True)

                # Log file has same name as directory + .quantize.log, and is placed alongside model directory, not inside it
                # This ensures that we can delete the output_dir in case of error or abort, without losing the logfile.
                # Therefore the existence of the output_dir is a reliable indicator of whether a model has started or not.
                logger.info(f"[{count} / {num_iters}] Quantizing: bits = {bits} - group_size = {group_size} - desc_act = {desc_act} - damp_percent = {damp} to {output_dir}")
                try:
                    quantize(args.pretrained_model_dir, output_dir, traindataset, bits, group_size, desc_act, damp, args.batch_size, args.use_triton, trust_remote_code=args.trust_remote_code, dtype=args.dtype)
                except KeyboardInterrupt:
                    logger.error(f"Aborted. Will delete {output_dir}")
                    os.rmdir(output_dir)
                    abort = True
                except:
                    raise

            finally:
                count += 1
        else:
                logger.error(f"Aborting - told to stop!")
                break