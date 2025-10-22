"""LM‑Fix module: polished for GitHub publishing."""
import torch
import torch.nn as nn
import hashlib

class GoldenHashGenerator:

    def __init__(self, model):
        self.model = model
        self.golden_hash = None
        self.last_linear_layer_name = None
        self.last_linear_layer = None

    def _compute_hash(self, tensor):
        tensor_np = tensor.cpu().to(torch.float32).numpy()
        return hashlib.sha256(tensor_np.tobytes()).hexdigest()

    def _hook_fn(self, module, input, output):
        self.golden_hash = self._compute_hash(output[0])

    def _find_last_linear_layer(self):
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
        if linear_layers:
            self.last_linear_layer_name, self.last_linear_layer = linear_layers[-1]
            return self.last_linear_layer
        else:
            raise RuntimeError('No Linear layers found.')

    def generate(self, input_ids_length, device):
        last_layer = self._find_last_linear_layer()
        hook = last_layer.register_forward_hook(self._hook_fn)
        input_ids = torch.arange(1, 1 + input_ids_length).unsqueeze(0).to(device)
        with torch.no_grad():
            self.model(input_ids=input_ids)
        hook.remove()
        return self.golden_hash

    def generate_with_previous_data(self, input_ids_length, device):
        last_layer = self.last_linear_layer
        hook = last_layer.register_forward_hook(self._hook_fn)
        input_ids = torch.arange(1, 1 + input_ids_length).unsqueeze(0).to(device)
        with torch.no_grad():
            self.model(input_ids=input_ids)
        hook.remove()
        return self.golden_hash