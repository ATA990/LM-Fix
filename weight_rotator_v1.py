"""LM‑Fix module: polished for GitHub publishing."""
import torch
import torch.nn as nn
import hashlib

class WeightRotator:

    def __init__(self, model, saved_outputs):
        self.model = model
        self.saved_outputs = saved_outputs
        self.rotated_outputs = {}
        self.layer_hashes = {}
        self.hooks = []

    def _compute_hash(self, tensor):
        tensor_np = tensor.cpu().detach().numpy()
        return hashlib.sha256(tensor_np.tobytes()).hexdigest()

    def _pre_hook(self, num_rows, start_value, steps):

        def hook(module, input):
            return (torch.linspace(start_value, start_value + (steps - 1) * start_value, steps=steps, dtype=input[0].dtype, device=input[0].device).unsqueeze(1).unsqueeze(2).expand(steps, 1, num_rows),)
        return hook

    def _forward_hook(self, layer_name):

        def hook(module, input, output):
            self.rotated_outputs[layer_name] = output.detach().clone()
            if isinstance(output, torch.Tensor):
                self.layer_hashes[layer_name] = self._compute_hash(output)
            if layer_name in self.saved_outputs:
                return self.saved_outputs[layer_name]
            return output
        return hook

    def _rotate_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight = nn.Parameter(torch.rot90(module.weight, k=-1, dims=[0, 1]))

    def _restore_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight = nn.Parameter(torch.rot90(module.weight, k=1, dims=[0, 1]))

    def register_hooks(self, start_value, steps):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                num_rows = module.weight.shape[1]
                self.hooks.append(module.register_forward_pre_hook(self._pre_hook(num_rows, start_value, steps)))
                self.hooks.append(module.register_forward_hook(self._forward_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def run(self, device, start_value=1000, steps=25):
        self._rotate_weights()
        self.register_hooks(start_value, steps)
        input_ids = torch.arange(1, 1 + steps).unsqueeze(1).to(device)
        with torch.no_grad():
            self.model(input_ids=input_ids)
        self.remove_hooks()
        self._restore_weights()
        return (self.rotated_outputs, self.layer_hashes)