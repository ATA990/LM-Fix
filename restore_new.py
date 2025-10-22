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
import hashlib
from golden_hash_generator import GoldenHashGenerator
from constant_input_hasher import ConstantInputHasher
from parameterized_input_saver import ParameterizedInputSaver
from weight_rotator import WeightRotator
import numpy as np

class ReStore:

    def __init__(self, model, tokenizer, ref_saved_outputs, ref_rotated_outputs, ref_layer_hashes, ref_rotated_layer_hashes, golden_hash, input_ids_length, start_value, steps, device, supported_parameters_per_layer=25):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_saved_outputs = ref_saved_outputs
        self.ref_rotated_outputs = ref_rotated_outputs
        self.ref_layer_hashes = ref_layer_hashes
        self.ref_rotated_layer_hashes = ref_rotated_layer_hashes
        self.golden_hash = golden_hash
        self.input_ids_length = input_ids_length
        self.start_value = start_value
        self.steps = steps
        self.device = device
        self.supported_parameters_per_layer = supported_parameters_per_layer
        self.generated_hash = None
        self.current_layer_hashes = {}
        self.current_saved_outputs = {}
        self.current_rotated_outputs = {}
        self.current_rotated_layer_hashes = {}

    def _compute_hash(self, tensor):
        tensor_np = tensor.cpu().detach().numpy()
        return hashlib.sha256(tensor_np.tobytes()).hexdigest()

    def generate_goldenHash(self):
        self.generated_hash = GoldenHashGenerator(self.model).generate(self.input_ids_length, self.device)

    def regenerate_and_collect(self):
        _log_print('\n🔄 Regenerating model outputs and hashes...')
        self.generated_hash = GoldenHashGenerator(self.model).generate(self.input_ids_length, self.device)
        self.current_layer_hashes = ConstantInputHasher(self.model).run(device=self.device, start_value=self.start_value, steps=self.steps)
        self.current_saved_outputs = ParameterizedInputSaver(self.model, self.supported_parameters_per_layer).run(device=self.device, start_value=self.start_value, steps=self.steps)
        self.current_rotated_outputs, self.current_rotated_layer_hashes = WeightRotator(self.model, self.ref_saved_outputs).run(device=self.device, start_value=self.start_value, steps=self.steps)
        _log_print('✅ Regeneration complete.\n')

    def update_current_saved_output(self):
        self.current_saved_outputs = ParameterizedInputSaver(self.model, self.supported_parameters_per_layer).run(device=self.device, start_value=self.start_value, steps=self.steps)

    def get_model_hash(self):
        self.generated_hash = GoldenHashGenerator(self.model).generate(self.input_ids_length, self.device)
        return self.generated_hash

    def regenerate_layers_hashes(self):
        _log_print('\nRegenerating model layers hashes...')
        self.current_layer_hashes = ConstantInputHasher(self.model).run(device=self.device, start_value=self.start_value, steps=self.steps)
        _log_print('Regeneration complete.\n')

    def compare_golden_hash(self):
        match = self.generated_hash == self.golden_hash
        _log_print(f"\nGolden Hash Match: {('YES' if match else 'NO')}")
        return match

    def compare_layer_hashes(self):
        corrupted_layers = []
        _log_print('\n🔍 Comparing layer hashes:')
        for layer_name, current_hash in self.current_layer_hashes.items():
            ref_hash = self.ref_layer_hashes.get(layer_name)
            if ref_hash and current_hash != ref_hash:
                _log_print(f'❌ {layer_name} - Hash mismatch')
                corrupted_layers.append(layer_name)
        _log_print(f'\nTotal Corrupted Layers: {len(corrupted_layers)}')
        return corrupted_layers

    def compare_layer_rotated_hashes(self):
        corrupted_layers = []
        _log_print('\nComparing layer hashes:')
        for layer_name, current_hash in self.current_rotated_layer_hashes.items():
            ref_hash = self.ref_rotated_layer_hashes.get(layer_name)
            if ref_hash and current_hash != ref_hash:
                _log_print(f'{layer_name} - Hash mismatch')
                corrupted_layers.append(layer_name)
        _log_print(f'\nTotal Corrupted Layers: {len(corrupted_layers)}')
        return corrupted_layers

    def generate_custom_matrix(self, offset, step, rows, cols):
        """
        Generates a matrix where each row has the same value.
        The value is calculated as: offset + row_index * step
        """
        A = [[offset + i * step for _ in range(cols)] for i in range(rows)]
        return A

    def solve_linear_system_lstsq(self, coeff_matrix, const_vector):
        """
        Solves the system of linear equations Ax = b using least squares.
        
        Parameters:
        coeff_matrix: list of lists or np.ndarray
        const_vector: list or np.ndarray
        
        Returns:
        x: Least squares solution
        residuals: Sum of residuals
        """
        A = np.array(coeff_matrix, dtype=float)
        b = np.array(const_vector, dtype=float)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return (x, residuals)

    def recover_model(self, faulty_parameters_x, faulty_parameters_y):
        for layer_name, x_set in faulty_parameters_x.items():
            y_set = faulty_parameters_y.get(layer_name)
            if len(x_set) == 1 and len(y_set) == 1:
                x = list(x_set)[0]
                y = list(y_set)[0]
                layer = self.model.get_submodule(layer_name)
                current_tensor = self.current_saved_outputs.get(layer_name)
                ref_tensor = self.ref_saved_outputs.get(layer_name)
                a = torch.tensor([self.start_value], dtype=torch.float32)
                s_old = torch.tensor([ref_tensor[0][0][x]], dtype=torch.float32)
                s_new = torch.tensor([current_tensor[0][0][x]], dtype=torch.float32)
                b = s_old - s_new
                resolved_value = b / a
                with torch.no_grad():
                    layer.weight[x][y] = resolved_value
                    _log_print(f'{layer_name} ({x},{y}) :  {resolved_value.to(dtype=torch.float32)}  Hex = {self.tensor_value_to_hex(value=resolved_value)}')
            elif len(x_set) > 0 and len(y_set) > 0:
                layer = self.model.get_submodule(layer_name)
                current_tensor = self.current_saved_outputs.get(layer_name)
                ref_tensor = self.ref_saved_outputs.get(layer_name)
                for x in x_set:
                    A = self.generate_custom_matrix(self.start_value, self.start_value, len(y_set), len(y_set))
                    b = []
                    i = 0
                    for y in y_set:
                        b.append(float(ref_tensor[0][i][x] - current_tensor[0][i][x]))
                        i += 1
                    solution, residuals = self.solve_linear_system_lstsq(A, b)
                    i = 0
                    for y in y_set:
                        with torch.no_grad():
                            layer.weight[x][y] = torch.tensor(solution[i], dtype=layer.weight.dtype)
                            _log_print(f'{layer_name} ({x},{y}) :  {solution[i]}')
                        i += 1

    def compare_saved_outputs(self, corrupted_layers):
        unequal_indices = {}
        _log_print('\n🔎 Comparing saved outputs with reference:')
        for layer_name in corrupted_layers:
            current_tensor = self.current_saved_outputs.get(layer_name)
            ref_tensor = self.ref_saved_outputs.get(layer_name)
            if current_tensor is None or ref_tensor is None:
                _log_print(f'[{layer_name}] Skipping - missing tensor')
                continue
            ref_vector = ref_tensor
            if ref_vector.shape != current_tensor.shape:
                _log_print(f'[{layer_name}] Skipping - shape mismatch')
                continue
            diff = current_tensor != ref_vector
            indices = torch.where(diff)
            index_list = list(zip(*[t.tolist() for t in indices]))
            unequal_indices[layer_name] = set((item[2] for item in index_list))
            _log_print(f'[{layer_name}] Unequal Indices: {unequal_indices[layer_name]}')
        return unequal_indices

    def compare_rotated_outputs(self, corrupted_layers):
        unequal_indices = {}
        _log_print('\n🔁 Comparing rotated outputs with reference:')
        for layer_name in corrupted_layers:
            current_tensor = self.current_rotated_outputs.get(layer_name)
            ref_tensor = self.ref_rotated_outputs.get(layer_name)
            if current_tensor is None or ref_tensor is None:
                _log_print(f'[{layer_name}] Skipping - missing tensor')
                continue
            ref_vector = ref_tensor
            if ref_vector.shape != current_tensor.shape:
                _log_print(f'[{layer_name}] Skipping - shape mismatch')
                continue
            diff = current_tensor != ref_vector
            indices = torch.where(diff)
            index_list = list(zip(*[t.tolist() for t in indices]))
            unequal_indices[layer_name] = set((item[2] for item in index_list))
            _log_print(f'[{layer_name}] Unequal Indices in Rotated Output: {unequal_indices[layer_name]}')
        _log_print(unequal_indices)
        return unequal_indices

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