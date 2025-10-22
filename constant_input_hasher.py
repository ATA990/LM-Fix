"""LM‑Fix module: polished for GitHub publishing."""
import torch
import torch.nn as nn
import hashlib

class ConstantInputHasher:

    def __init__(self, model):
        self.model = model
        self.layer_hashes = {}
        self.layer_entire_hash = {}
        self.layer_sum_out = {}
        self.layer_outs = {}
        self.hooks = []

    def _compute_hash(self, tensor):
        tensor_np = tensor.cpu().to(torch.float32).numpy()
        return hashlib.sha256(tensor_np.tobytes()).hexdigest()

    def _pre_hook(self, num_rows, start_value, steps):

        def hook(module, input):
            return (torch.linspace(start_value, start_value + (steps - 1) * start_value, steps=steps, dtype=input[0].dtype, device=input[0].device).unsqueeze(0).unsqueeze(2).expand(1, steps, num_rows),)
        return hook

    def _forward_hook(self, layer_name, out_key):

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                sum_out = output[0].to(torch.float64).sum(dim=0)
                self.layer_hashes[layer_name] = self._compute_hash(sum_out)
                self.layer_sum_out[layer_name] = sum_out
                if out_key not in self.layer_outs:
                    self.layer_outs[out_key] = output
        return hook

    def register_hooks(self, start_value, steps):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                num_rows = module.weight.shape[1]
                self.hooks.append(module.register_forward_pre_hook(self._pre_hook(num_rows, start_value, steps)))
                self.hooks.append(module.register_forward_hook(self._forward_hook(name, module.weight.shape[0])))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def run(self, device, start_value=1000, steps=25):
        self.register_hooks(start_value, steps)
        input_ids = torch.arange(1, 1 + steps).unsqueeze(0).to(device)
        with torch.no_grad():
            self.model(input_ids=input_ids)
        self.remove_hooks()
        return (self.layer_hashes, self.layer_sum_out, self.layer_entire_hash, self.layer_outs)