logger = logging.getLogger(__name__)

def _log_print(*args, **kwargs):
    try:
        msg = ' '.join((str(a) for a in args))
    except Exception:
        msg = ' '.join(map(str, args))
    logger.info(msg)
import logging
'LM‑Fix module: polished for GitHub publishing.'
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from os.path import isfile
from huggingface_hub import login
import numpy as np
import random
from datetime import datetime
import statistics
import csv
import sys
import time
from models.predefined_model_creators import predefined_model_creators
from llm_bfa.helpers.functions import handleQuantized
import hashlib
import general_functions as gf
import torch.nn.functional as F
import datasets
import time
from tqdm import tqdm
from golden_hash_generator import GoldenHashGenerator
login('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')
log_folder = '/home/atahmasi/BFA-Defence/01-Framework/LogData'
TestVectorLens = [1, 10, 40, 100, 200, 600, 1000]
arguments = sys.argv
model_name = 'QwQ-32B-FP8-dynamic'
if len(arguments) > 1:
    model_name = arguments[1]
    _log_print(f'modelName: {model_name}')
LogFileName = f'{log_folder}/Overhead/{model_name}.txt'
time.sleep(1)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    _log_print('Checking GPU befor running')
    _log_print(f'CUDA is available! GPU: {torch.cuda.get_device_name(0)}')
    _log_print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB')
    _log_print(f'GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB')
else:
    _log_print('L CUDA is NOT available. Running on CPU.')
_log_print('Loading model')
model_creator = predefined_model_creators[model_name]
tokenizer = model_creator.load_tokenizer()
model = model_creator.load_model()
handleQuantized(model)
if torch.cuda.is_available():
    _log_print(f'Analizing Allocated CUDA Mem is available! GPU: {torch.cuda.get_device_name(0)}')
    _log_print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB')
    _log_print(f'GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB')
now = datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
text_to_add = f'\n\n\n{timestamp} - {model_name} Dtype = {model.model.layers[0].self_attn.o_proj.weight.dtype} \n'
with open(LogFileName, 'a') as file:
    file.write(text_to_add)
GHash = GoldenHashGenerator(model)
detectionTimes = {}
for count in TestVectorLens:
    start_time = time.perf_counter()
    generated_hash = GHash.generate(input_ids_length=count, device=device)
    end_time = time.perf_counter()
    totalTime1 = end_time - start_time
    start_time = time.perf_counter()
    generated_hash2 = GHash.generate_with_previous_data(input_ids_length=count, device=device)
    end_time = time.perf_counter()
    totalTime2 = end_time - start_time
    start_time = time.perf_counter()
    generated_hash3 = GHash.generate_with_previous_data(input_ids_length=count, device=device)
    end_time = time.perf_counter()
    totalTime3 = end_time - start_time
    detectionTimes[count] = [totalTime1, totalTime2, totalTime3]
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    text_to_add = f'{timestamp} - Detection time: Test Vector lenght: {count} - T1:{totalTime1}s T2:{totalTime2}s T3:{totalTime3}s\n'
    with open(LogFileName, 'a') as file:
        file.write(text_to_add)
input_texts = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text'][:10]
input_texts = [s for s in input_texts if s != '']
totalGenerationCounts = len(input_texts)
_log_print(f'total input text count = {totalGenerationCounts}')
start_time = time.perf_counter()
for text in tqdm(input_texts, desc='Evaluating Latancy using wikitext-2-raw-v1'):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=True, min_new_tokens=150)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
end_time = time.perf_counter()
totalTime_generation = end_time - start_time
now = datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
_log_print('Wiki-text uprotected Elapsed time:', totalTime_generation, 'seconds')
text_to_add = f'{timestamp} - Wiki-text uprotected Elapsed time: {totalTime_generation}seconds\n'
with open(LogFileName, 'a') as file:
    file.write(text_to_add)
for count in TestVectorLens:
    detectionTime = detectionTimes[count][0] + (detectionTimes[count][1] + detectionTimes[count][2]) / 2 * (totalGenerationCounts - 1)
    overhead = detectionTime / (totalTime_generation + detectionTime)
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    _log_print(f'Performance overhead for text vector {count} = {overhead * 100} ')
    text_to_add = f'{timestamp} - Performance overhead for text vector {count} = {overhead * 100} \n'
    with open(LogFileName, 'a') as file:
        file.write(text_to_add)