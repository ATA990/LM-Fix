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
import torch.nn as nn
import random
import numpy as np
import general_functions as gf
from transformers import AutoModelForCausalLM, AutoTokenizer

class BitFlipInjector:

    def __init__(self, model, total_flips=10, bit_flip_mode='random'):
        """
        Initializes the BitFlipInjector class.

        Args:
            model (torch.nn.Module): The PyTorch model.
            total_flips (int): Total number of bit flips to perform.
        """
        self.model = model
        self.total_flips = total_flips
        self.log = []
        self.linear_layers = self._find_linear_layers()
        self.bit_flip_mode = bit_flip_mode

    def _find_linear_layers(self):
        """Finds all Linear layers in the model."""
        return {name: module for name, module in self.model.named_modules() if isinstance(module, nn.Linear)}

    def float_to_hex(self, value, dtype):
        """
        Converts a floating-point or integer value to its hexadecimal representation.
        """
        if dtype.is_floating_point or dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return hex(np.frombuffer(np.array(value, dtype=dtype.numpy().dtype).tobytes(), dtype=np.uint64)[0])
        elif dtype in [torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5, torch.uint6, torch.uint7, torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            return hex(value)
        elif dtype in [torch.qint8, torch.qint32, torch.quint2x4, torch.quint4x2, torch.quint8]:
            return hex(value.q_per_channel_scales().item())
        else:
            return 'Unsupported dtype'

    def tensor_value_to_hex(self, value: torch.Tensor) -> str:
        """
        Converts a scalar tensor (0D) to its IEEE-754 or integer hex representation.

        Args:
            value (torch.Tensor): A scalar tensor (with or without requires_grad).

        Returns:
            str: Hexadecimal string.
        """
        assert value.numel() == 1, 'Only scalar tensors are supported.'
        value = value.detach().cpu()
        dtype = value.dtype
        if dtype in [torch.bfloat16]:
            value = value.to(dtype=torch.float32)
        if dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]:
            return hex(value.view(torch.uint8).item())
        raw_bytes = value.numpy().tobytes()
        int_value = int.from_bytes(raw_bytes, byteorder='little')
        return hex(int_value)

    def flip_one_bit(self):
        """
        Flips exactly one random bit in a randomly selected Linear layer.
        """
        if not self.linear_layers:
            _log_print('No Linear layers found in the model.')
            return
        filtered_layers = {name: layer for name, layer in self.linear_layers.items() if 'ali' not in name}
        selected_layer_name, selected_layer = random.choice(list(filtered_layers.items()))
        while selected_layer.weight.device != 'cuda' or selected_layer.weight.device != 'cpu':
            selected_layer_name, selected_layer = random.choice(list(filtered_layers.items()))
        with torch.no_grad():
            dtype = selected_layer.weight.dtype
            device = selected_layer.weight.device
            Target_weights = selected_layer.weight
            bit_width_map = {torch.float8_e4m3fn: 8, torch.float8_e4m3fnuz: 8, torch.float8_e5m2: 8, torch.float8_e5m2fnuz: 8, torch.float16: 16, torch.float32: 32, torch.float64: 64, torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64, torch.uint1: 1, torch.uint2: 2, torch.uint3: 3, torch.uint4: 4, torch.uint5: 5, torch.uint6: 6, torch.uint7: 7, torch.uint8: 8, torch.uint16: 16, torch.uint32: 32, torch.uint64: 64, torch.qint8: 8, torch.qint32: 32, torch.quint2x4: 8, torch.quint4x2: 8, torch.quint8: 8, torch.bfloat16: 16}
            if dtype in bit_width_map:
                bit_width = bit_width_map[dtype]
            else:
                _log_print(f'Unsupported dtype: {dtype}')
                return
            dimXWeight = len(Target_weights)
            dimYWeight = len(Target_weights[0])
            posXparam = random.randint(0, dimXWeight - 1)
            posYparam = random.randint(0, dimYWeight - 1)
            bit_position = random.randint(0, bit_width - 1)
            old_value_float = Target_weights[posXparam][posYparam].clone().detach()
            old_value_hex = self.tensor_value_to_hex(value=old_value_float)
            modifiedParam = gf.tensor_flip_bit(Target_weights[posXparam][posYparam], bit_position, dtype)
            Target_weights[posXparam][posYparam] = modifiedParam.clone().detach()
            new_value_float = Target_weights[posXparam][posYparam].clone().detach()
            new_value_hex = self.tensor_value_to_hex(new_value_float)
            param_coords = f'({posXparam},{posYparam})'
            self.log.append({'Layer': selected_layer_name, 'Parameter Index': param_coords, 'Bit Position': bit_position, 'Old Value (Float)': old_value_float.to(dtype=torch.float32), 'Old Value (org)': old_value_float, 'Old Value (Hex)': old_value_hex, 'New Value (Float)': new_value_float.to(dtype=torch.float32), 'New Value (Hex)': new_value_hex, 'Data Type': str(dtype)})

    def flip_one_bit_targeted(self, selected_layer_name, param_x, param_y, bit_location):
        """
        Flips exactly one random bit in a randomly selected Linear layer.
        """
        if not self.linear_layers:
            _log_print('No Linear layers found in the model.')
            return
        selected_layer = self.model.get_submodule(selected_layer_name)
        with torch.no_grad():
            dtype = selected_layer.weight.dtype
            device = selected_layer.weight.device
            Target_weights = selected_layer.weight
            posXparam = param_x
            posYparam = param_y
            bit_position = bit_location
            old_value_float = Target_weights[posXparam][posYparam].clone().detach()
            old_value_hex = self.tensor_value_to_hex(value=old_value_float)
            modifiedParam = gf.tensor_flip_bit(Target_weights[posXparam][posYparam], bit_position, dtype)
            Target_weights[posXparam][posYparam] = modifiedParam.clone().detach()
            new_value_float = Target_weights[posXparam][posYparam].clone().detach()
            new_value_hex = self.tensor_value_to_hex(new_value_float)
            param_coords = f'({posXparam},{posYparam})'
            self.log.append({'Layer': selected_layer_name, 'Parameter Index': param_coords, 'Bit Position': bit_position, 'Old Value (Float)': old_value_float.to(dtype=torch.float32), 'Old Value (org)': old_value_float, 'Old Value (Hex)': old_value_hex, 'New Value (Float)': new_value_float.to(dtype=torch.float32), 'New Value (Hex)': new_value_hex, 'Data Type': str(dtype)})

    def inject_faults(self):
        """
        Injects bit-flip faults one at a time, up to the specified total number of flips.
        Returns:
            dict: A dictionary mapping layer names to lists of parameter indices where faults were injected.
        """
        self.log.clear()
        injection_summary = {}
        for _ in range(self.total_flips):
            self.flip_one_bit()
            entry = self.log[-1]
            layer = entry['Layer']
            param_index = eval(entry['Parameter Index'])
            if layer not in injection_summary:
                injection_summary[layer] = []
            injection_summary[layer].append(param_index)
        return (injection_summary, self.log)

    def inject_Targeted_faults(self, Tagets):
        """
        Injects bit-flip faults one at a time, up to the specified total number of flips.
        Returns:
            dict: A dictionary mapping layer names to lists of parameter indices where faults were injected.
        """
        self.log.clear()
        injection_summary = {}
        for layer_name, pos_x, pos_y, bit_loc in Tagets:
            self.flip_one_bit_targeted(layer_name, pos_x, pos_y, bit_loc)
            entry = self.log[-1]
            layer = entry['Layer']
            param_index = eval(entry['Parameter Index'])
            if layer not in injection_summary:
                injection_summary[layer] = []
            injection_summary[layer].append(param_index)
        return (injection_summary, self.log)

    def print_log(self):
        """
        Prints the log of bit flips.
        """
        if not self.log:
            _log_print('No bit flips were applied.')
            return
        _log_print('\nBit Flip Log:')
        for entry in self.log:
            _log_print(f"Layer: {entry['Layer']}, Parameter Index: {entry['Parameter Index']}, Bit Position: {entry['Bit Position']}, Old Value: {entry['Old Value (Float)']} ({entry['Old Value (Hex)']}), New Value: {entry['New Value (Float)']} ({entry['New Value (Hex)']}), Data Type: {entry['Data Type']}")

    def reStore_by_last(self):
        for log in self.log:
            layerName = log['Layer']
            param_index = eval(log['Parameter Index'])
            layer = self.model.get_submodule(layerName)
            x = param_index[0]
            y = param_index[1]
            with torch.no_grad():
                if layer.weight.dtype == torch.int8:
                    layer.weight[x][y] = log['Old Value (org)']
                else:
                    layer.weight[x][y] = nn.Parameter(log['Old Value (org)'])