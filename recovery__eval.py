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
from golden_hash_generator import GoldenHashGenerator
from constant_input_hasher import ConstantInputHasher as RefrenceHasher
from constant_input_hasher_tensor import ConstantInputHasher as RecoveryHasher
from parameterized_input_saver_v2_1 import ParameterizedInputSaver
from weight_rotator_v2 import WeightRotator
import os
import torch.nn as nn
import time
from pympler import asizeof
import sys

class ReferencePipeline:

    def __init__(self, model, tokenizer):
        """
        Initializes the reference pipeline with an external model and tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def run(self, input_ids_length, start_value, layer_steps, device):
        """
        Executes all four steps of the pipeline.
        
        Parameters:
        - input_ids_length: Number of input tokens used for golden hash
        - start_value: Starting value for constant/parameterized input vectors
        - steps: Number of steps for vector generation
        - device: "cuda" or "cpu"
        """
        _log_print('\n=== Step 1: Generating Golden Hash ===')
        golden_gen = GoldenHashGenerator(self.model)
        golden_hash = golden_gen.generate(input_ids_length, device)
        _log_print('\n=== Step 2: Constant Input Layer Hashing ===')
        hasher = RefrenceHasher(self.model)
        layer_hashes, layer_sum_out, layer_entire_hash, layer_outs = hasher.run(device=device, start_value=start_value, steps=layer_steps)
        _log_print('\n=== Step 3: Saving Outputs with Parameterized Inputs ===')
        saver = ParameterizedInputSaver(self.model)
        saved_outputs, base_vector = saver.run(device=device)
        _log_print('\n=== Step 4: Rotating Weights and Comparing Outputs ===')
        rotator = WeightRotator(self.model, layer_outs)
        rotated_outputs, rotated_layer_hashes = rotator.run(device=device, start_value=1e-06, steps=layer_steps)
        return {'golden_hash': golden_hash, 'layer_hashes': layer_hashes, 'saved_outputs': saved_outputs, 'rotated_outputs': rotated_outputs, 'rotated_layer_hashes': rotated_layer_hashes, 'layer_sum_out': layer_sum_out, 'layer_entire_hash': layer_entire_hash, 'layer_outs': layer_outs, 'base_vector': base_vector}

def recursive_delete(obj):
    if hasattr(obj, '__dict__'):
        for key in list(vars(obj)):
            try:
                setattr(obj, key, None)
            except Exception:
                pass
    if hasattr(obj, 'modules'):
        for child in obj.modules():
            recursive_delete(child)

def remove_all_hooks(model):
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()

def full_model_cleanup(model, tokenizer=None):
    remove_all_hooks(model)
    recursive_delete(model)
    if tokenizer:
        recursive_delete(tokenizer)
    del model
    if tokenizer:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    _log_print('Model and all references removed from memory.')
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from restore_new_v2 import ReStore
    from bit_flip_injector import BitFlipInjector
    from huggingface_hub import login
    login('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')
    from models.predefined_model_creators import predefined_model_creators
    from llm_bfa.helpers.functions import handleQuantized
    import gc
    import time
    import csv
    input_ids_length = 200
    start_value = 1
    layer_steps = 1000
    arguments = sys.argv
    model_name = 'Llama-3.2-1B-Instruct'
    model_name = 'Llama-3.2-3B-Instruct'
    model_name = 'Llama-3.2-3B-Instruct-W8'
    model_name = 'QwQ-32B-FP8-dynamic'
    BF_count = 5
    if len(arguments) > 1:
        model_name = arguments[1]
        _log_print(f'modelName: {model_name}')
    if len(arguments) > 2:
        BF_count = int(arguments[2])
        _log_print(f'BF count: {BF_count}')
    log_name = model_name
    totalIterations = 1
    faultList = []
    faultList.append(BF_count)
    modelSavePath = f'/home/atahmasi/Desktop/SCRATCH/LLMs_Layers/{model_name}'
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)
    device = 'cpu'
    _log_print('Loading model')
    model_creator = predefined_model_creators[model_name]
    model_loading_time_cloud = 0
    if torch.cuda.is_available():
        device = 'cuda'
        _log_print('Checking GPU befor running')
        _log_print(f'CUDA is available! GPU: {torch.cuda.get_device_name(0)}')
        _log_print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB')
        _log_print(f'GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB')
    else:
        _log_print('L CUDA is NOT available. Running on CPU.')
    start = time.perf_counter()
    model = model_creator.load_model()
    end = time.perf_counter()
    tokenizer = model_creator.load_tokenizer()
    handleQuantized(model)
    model_loading_time_Hdd = end - start
    memory_usage = 0
    if torch.cuda.is_available():
        _log_print(f'Analizing Allocated CUDA Mem is available! GPU: {torch.cuda.get_device_name(0)}')
        _log_print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB')
        _log_print(f'GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB')
        memory_usage = torch.cuda.memory_reserved(0)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            num_rows = module.weight.shape[1]
            torch.save(module.weight, f'{modelSavePath}/{name}.pt')
    pipeline = ReferencePipeline(model, tokenizer)
    result = pipeline.run(input_ids_length, start_value, layer_steps, device)
    memory_Overhead = asizeof.asizeof(result)
    _log_print(f'Memory Ovehead = {memory_Overhead / 1024 ** 2:.2f} MB')
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    _log_print('\nFinal Golden Hash:', result['golden_hash'])
    golden_hash = result['golden_hash']
    ref_layer_hashes = result['layer_hashes']
    ref_saved_outputs = result['saved_outputs']
    ref_rotated_outputs = result['rotated_outputs']
    ref_rotated_layer_hashes = result['rotated_layer_hashes']
    ref_layer_sum_out = result['layer_sum_out']
    ref_layer_entire_hash = result['layer_entire_hash']
    ref_layer_outs_ref_for_rotation = result['layer_outs']
    ref_base_vector = result['base_vector']
    LogFileName = f'/home/atahmasi/BFA-Defence/01-Framework/LogData/Recovery/{log_name}__{BF_count}.csv'
    log_header = []
    log_header.append('Model_load_time_Hdd')
    log_header.append('Model_load_time_cloud')
    log_header.append('model_memory')
    log_header.append('overhead_memory')
    log_header.append('injected_faults')
    log_header.append('detection_status')
    log_header.append('recovery_status')
    log_header.append('overhead_time')
    with open(LogFileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(log_header)
    for faultCount in faultList:
        average_overhead = 0
        sum_overhead = 0
        for k in range(1, totalIterations + 1):
            Log = []
            Log.append(model_loading_time_Hdd)
            Log.append(model_loading_time_cloud)
            Log.append(memory_usage)
            Log.append(memory_Overhead)
            Log.append(faultCount)
            injector = BitFlipInjector(model, total_flips=faultCount)
            faults, log = injector.inject_faults()
            restorer = ReStore(model=model, tokenizer=tokenizer, ref_saved_outputs=ref_saved_outputs, ref_rotated_outputs=ref_rotated_outputs, ref_layer_hashes=ref_layer_hashes, ref_rotated_layer_hashes=ref_rotated_layer_hashes, golden_hash=golden_hash, input_ids_length=input_ids_length, start_value=start_value, layer_steps=layer_steps, device=device, ref_layer_sum_out=ref_layer_sum_out, ref_layer_entire_hash=ref_layer_entire_hash, ref_layer_outs_ref_for_rotation=ref_layer_outs_ref_for_rotation, ref_base_vector=ref_base_vector)
            restorer.get_model_hash()
            match = restorer.compare_golden_hash()
            Log.append('ok' if match else 'fail')
            start = time.perf_counter()
            restorer.regenerate_layers_hashes()
            corrupted_layers, faulty_parameters_x = restorer.compare_layer_hashes()
            _, faulty_parameters_y = restorer.find_crrupted_params_y(corrupted_layers)
            loging = True
            try:
                restorer.recover_model(faulty_parameters_x=faulty_parameters_x, faulty_parameters_y=faulty_parameters_y)
            except Exception:
                loging = False
                _log_print('Error In recovery')
            end = time.perf_counter()
            restorer.get_model_hash()
            match = restorer.compare_golden_hash()
            Log.append('ok' if match else 'fail')
            recovery_time = end - start
            Log.append(recovery_time)
            if loging:
                with open(LogFileName, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(Log)
            else:
                k -= 1
                for layerName, positions in faults.items():
                    _log_print(f'  {layerName}')
                    layer = model.get_submodule(layerName)
                    loaded_tensor = torch.load(f'{modelSavePath}/{layerName}.pt', weights_only=True).to(layer.weight.device)
                    layer.weight.data.copy_(loaded_tensor)
                restorer.get_model_hash()
                match = restorer.compare_golden_hash()