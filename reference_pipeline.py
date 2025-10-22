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
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from restore_new import ReStore
    from bit_flip_injector import BitFlipInjector
    model_name = 'meta-llama/Llama-3.2-3B'
    input_ids_length = 200
    start_value = 1
    steps = 200
    device = 'cuda'
    supported_parameters_per_layer = steps
    modelSavePath = f'/home/atahmasi/Desktop/SCRATCH/{model_name}'
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    end = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_loading_time = end - start
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            num_rows = module.weight.shape[1]
            torch.save(module.weight, f'{modelSavePath}/{name}.pt')
    pipeline = ReferencePipeline(model, tokenizer, supported_parameters_per_layer=steps)
    result = pipeline.run(input_ids_length, start_value, steps, device)
    _log_print('\nFinal Golden Hash:', result['golden_hash'])
    injector = BitFlipInjector(model, total_flips=1)
    faults, log = injector.inject_faults()
    injector.print_log()
    _log_print('\nInjected Fault Summary:')
    unique_layer_count = 0
    for layer, positions in faults.items():
        unique_layer_count += 1
        _log_print(f'\n{unique_layer_count}-Layer: {layer}')
        for pos in positions:
            _log_print(f'  Injected Fault at: {pos}')
    _log_print(f'\nTotal Number of Faulty layers: {unique_layer_count}')
    golden_hash = result['golden_hash']
    ref_layer_hashes = result['layer_hashes']
    ref_saved_outputs = result['saved_outputs']
    ref_rotated_outputs = result['rotated_outputs']
    ref_rotated_layer_hashes = result['rotated_layer_hashes']
    restorer = ReStore(model=model, tokenizer=tokenizer, ref_saved_outputs=ref_saved_outputs, ref_rotated_outputs=ref_rotated_outputs, ref_layer_hashes=ref_layer_hashes, ref_rotated_layer_hashes=ref_rotated_layer_hashes, golden_hash=golden_hash, input_ids_length=input_ids_length, start_value=start_value, steps=steps, device=device, supported_parameters_per_layer=supported_parameters_per_layer)
    restorer.regenerate_and_collect()
    restorer.compare_golden_hash()
    corrupted_layers = restorer.compare_layer_hashes()
    faulted_layer_names = set(faults.keys())
    expected_corrupted = set(corrupted_layers)
    missing_layers = faulted_layer_names - expected_corrupted
    _log_print('\nLayers with injected faults but missing in corrupted_layers:')
    _log_print(f'\nTotal missing layers: {len(missing_layers)}')
    start = time.perf_counter()
    for layerName in missing_layers:
        _log_print(f'  {layerName}')
    end = time.perf_counter()
    reloading_time = end - start
    faulty_parameters_x = restorer.compare_saved_outputs(corrupted_layers)
    faulty_parameters_y = restorer.compare_rotated_outputs(corrupted_layers)
    for layer_name, x_set in faulty_parameters_x.items():
        y_set = faulty_parameters_y.get(layer_name)
        if len(x_set) > 0 and len(y_set) > 0:
            layer = model.get_submodule(layer_name)
            for x in x_set:
                for y in y_set:
                    with torch.no_grad():
                        layer.weight[x][y] = torch.tensor(0.0, dtype=layer.weight.dtype)
    restorer.update_current_saved_output()
    restorer.recover_model(faulty_parameters_x=faulty_parameters_x, faulty_parameters_y=faulty_parameters_y)
    restorer.generate_goldenHash()
    restorer.compare_golden_hash()