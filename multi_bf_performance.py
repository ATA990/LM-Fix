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
from bit_flip_injector import BitFlipInjector
import hashlib
import general_functions as gf
import torch.nn.functional as F
import datasets
import time
from tqdm import tqdm
from golden_hash_generator import GoldenHashGenerator
login('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')
log_folder = '/home/atahmasi/BFA-Defence/01-Framework/LogData'
TestVectorLens = [1, 10, 40]
bf_counts = [1, 2, 5, 10, 20, 35]
arguments = sys.argv
model_name = 'Qwen2-7B-Instruct'
IterationCount = 20

def disable_input_activation_quantization(module):
    for name, child in module.named_children():
        if child.__class__.__name__ == 'CompressedLinear':
            child.quantization_scheme.input_activations = None
        else:
            disable_input_activation_quantization(child)
if len(arguments) > 1:
    model_name = arguments[1]
    _log_print(f'modelName: {model_name}')
if len(arguments) > 2:
    IterationCount = int(arguments[2])
    _log_print(f'IterationCount: {IterationCount}')
filename = 'Llama-3.2-3B-Instruct-FP8'
LogFileName = f'{log_folder}/Multi_BF_Performance/{model_name}.txt'
csvLogFileName = f'{log_folder}/Multi_BF_Performance/{model_name}.csv'
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
log_header = []
log_header.append('TVL')
log_header.append('Layer')
log_header.append('TPL 1')
log_header.append('TBL 1')
log_header.append('value_befor')
log_header.append('value_after')
log_header.append('Detection')
log_header.append('executionTime')
with open(csvLogFileName, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(log_header)
now = datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
text_to_add = f'\n\n\n{timestamp} - {model_name} Dtype = {model.model.layers[0].self_attn.o_proj.weight.dtype} \n'
with open(LogFileName, 'a') as file:
    file.write(text_to_add)
GHash = GoldenHashGenerator(model)
detectionTimes = {}
for bf_count in bf_counts:
    injector = BitFlipInjector(model, total_flips=bf_count)
    for count in tqdm(TestVectorLens, desc='Performance Evaluation - BF_Count = {}'):
        start_time = time.perf_counter()
        Golden_hash = GHash.generate(input_ids_length=count, device=device)
        end_time = time.perf_counter()
        totalTime1 = end_time - start_time
        failCount = 0
        for i in tqdm(range(IterationCount), desc=f'TVL = {count}'):
            injection_Log = []
            injection_Log.append(count)
            faults, log = injector.inject_faults()
            injection_Log.append(log[0]['Layer'])
            injection_Log.append(log[0]['Parameter Index'])
            injection_Log.append(log[0]['Bit Position'])
            injection_Log.append(log[0]['Old Value (Hex)'])
            injection_Log.append(log[0]['New Value (Hex)'])
            start_time = time.perf_counter()
            generated_hash2 = GHash.generate_with_previous_data(input_ids_length=count, device=device)
            end_time = time.perf_counter()
            totalTime2 = end_time - start_time
            if generated_hash2 == Golden_hash:
                injection_Log.append('fail')
                failCount += 1
            else:
                injection_Log.append('ok')
            injection_Log.append(totalTime1)
            with open(csvLogFileName, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(injection_Log)
            injector.reStore_by_last()
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        text_to_add = f'{timestamp} - BF_Count = {bf_count} - TVL = {count}, Itration:{IterationCount}, Fails:{(failCount,)} Coverage = {(IterationCount - failCount) / IterationCount * 100} \n'
        with open(LogFileName, 'a') as file:
            file.write(text_to_add)