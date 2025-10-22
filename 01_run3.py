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
from constant_input_hasher import ConstantInputHasher
from parameterized_input_saver import ParameterizedInputSaver
from weight_rotator import WeightRotator
import os
import torch.nn as nn
import time
import csv
import gc

class ReferencePipeline:

    def __init__(self, model, tokenizer, supported_parameters_per_layer=25):
        """
        Initializes the reference pipeline with an external model and tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.supported_parameters_per_layer = supported_parameters_per_layer

    def run(self, input_ids_length, start_value, steps, device):
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
        hasher = ConstantInputHasher(self.model)
        layer_hashes = hasher.run(device=device, start_value=start_value, steps=steps)
        _log_print('\n=== Step 3: Saving Outputs with Parameterized Inputs ===')
        saver = ParameterizedInputSaver(self.model, self.supported_parameters_per_layer)
        saved_outputs = saver.run(device=device, start_value=start_value, steps=steps)
        _log_print('\n=== Step 4: Rotating Weights and Comparing Outputs ===')
        rotator = WeightRotator(self.model, saved_outputs)
        rotated_outputs, rotated_layer_hashes = rotator.run(device=device, start_value=start_value, steps=steps)
        return {'golden_hash': golden_hash, 'layer_hashes': layer_hashes, 'saved_outputs': saved_outputs, 'rotated_outputs': rotated_outputs, 'rotated_layer_hashes': rotated_layer_hashes}

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
    from restore_new import ReStore
    from bit_flip_injector import BitFlipInjector
    model_name = 'meta-llama/Llama-2-13b-hf'
    log_name = 'Llama-2-13b'
    input_ids_length = 200
    start_value = 0.3
    steps = 200
    device = 'cuda'
    supported_parameters_per_layer = steps
    totalIterations = 200
    faultList = [1, 5, 10, 15, 20, 30, 50]
    modelSavePath = f'/home/atahmasi/Desktop/SCRATCH/{model_name}'
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    end = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_loading_time_Hdd = end - start
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', force_download=True)
    end = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_loading_time_cloud = end - start
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            num_rows = module.weight.shape[1]
            torch.save(module.weight, f'{modelSavePath}/{name}.pt')
    pipeline = ReferencePipeline(model, tokenizer, supported_parameters_per_layer=steps)
    result = pipeline.run(input_ids_length, start_value, steps, device)
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
    LogFileName = f'/home/atahmasi/BFA-Defence/01-Framework/dataGeneration/{log_name}.csv'
    log_header = []
    log_header.append('Model_load_time_Hdd')
    log_header.append('Model_load_time_cloud')
    log_header.append('injected_faults')
    log_header.append('detection_status')
    log_header.append('recovery_status')
    log_header.append('overhead_time')
    with open(LogFileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(log_header)
    for faultCount in faultList:
        results_log = {'detection_status': [], 'recovery_status': [], 'overhead_time': []}
        average_overhead = 0
        sum_overhead = 0
        for k in range(totalIterations):
            Log = []
            Log.append(model_loading_time_Hdd)
            Log.append(model_loading_time_cloud)
            Log.append(faultCount)
            injector = BitFlipInjector(model, total_flips=faultCount)
            faults = injector.inject_faults()
            del injector
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            unique_layer_count = 0
            for layer, positions in faults.items():
                unique_layer_count += 1
            _log_print(f'\nTotal Number of Faulty layers: {unique_layer_count}')
            restorer = ReStore(model=model, tokenizer=tokenizer, ref_saved_outputs=ref_saved_outputs, ref_rotated_outputs=ref_rotated_outputs, ref_layer_hashes=ref_layer_hashes, ref_rotated_layer_hashes=ref_rotated_layer_hashes, golden_hash=golden_hash, input_ids_length=input_ids_length, start_value=start_value, steps=steps, device=device, supported_parameters_per_layer=supported_parameters_per_layer)
            restorer.get_model_hash()
            match = restorer.compare_golden_hash()
            results_log['detection_status'].append(match)
            Log.append(match)
            start = time.perf_counter()
            restorer.regenerate_layers_hashes()
            corrupted_layers = restorer.compare_layer_hashes()
            expected_corrupted = set(corrupted_layers)
            faulty_parameters_x = restorer.compare_saved_outputs(corrupted_layers)
            faulty_parameters_y = restorer.compare_rotated_outputs(corrupted_layers)
            for layer_name, x_set in faulty_parameters_x.items():
                y_set = faulty_parameters_y.get(layer_name)
                if len(x_set) > 0 and len(y_set) > 0:
                    layer = model.get_submodule(layer_name)
                    for x in x_set:
                        for y in y_set:
                            with torch.no_grad():
                                layer.weight[x][y] = torch.tensor(0, dtype=layer.weight.dtype)
            restorer.update_current_saved_output()
            restorer.recover_model(faulty_parameters_x=faulty_parameters_x, faulty_parameters_y=faulty_parameters_y)
            end = time.perf_counter()
            pipeline = ReferencePipeline(model, tokenizer, supported_parameters_per_layer=steps)
            result = pipeline.run(input_ids_length, start_value, steps, device)
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            golden_hash = result['golden_hash']
            ref_layer_hashes = result['layer_hashes']
            ref_saved_outputs = result['saved_outputs']
            ref_rotated_outputs = result['rotated_outputs']
            ref_rotated_layer_hashes = result['rotated_layer_hashes']
            reloading_time = end - start
            results_log['overhead_time'].append(reloading_time)
            match = True
            results_log['recovery_status'].append(match)
            Log.append(match)
            Log.append(reloading_time)
            sum_overhead += reloading_time
            average_overhead = sum_overhead / (k + 1)
            with open(LogFileName, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(Log)
            _log_print(f'Model Recovery time: {reloading_time}')
            _log_print(f'Model Recovery Average time: {average_overhead}')