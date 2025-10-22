"""LM‑Fix module: polished for GitHub publishing."""
import torch
import torch.nn as nn
import hashlib
from SimpleLinearModel import SimpleLinearModel
import gc

class WeightRotator:

    def __init__(self, model, saved_outputs):
        self.model = model
        self.saved_outputs = saved_outputs
        self.rotated_outputs = {}
        self.layer_hashes = {}
        self.hooks = []
        self.steps = 30

    def _compute_hash(self, tensor):
        tensor_np = tensor.cpu().to(torch.float32).numpy()
        return hashlib.sha256(tensor_np.tobytes()).hexdigest()

    def _pre_hook(self, num_rows, start_value, steps):

        def hook(module, input):
            input = (torch.linspace(start_value, start_value + (steps - 1) * start_value, steps=steps, dtype=input[0].dtype, device=input[0].device).unsqueeze(1).unsqueeze(2).expand(steps, 1, num_rows),)
            return input
        return hook

    def _forward_hook(self, layer_name, out_key):

        def hook(module, input, output):
            self.rotated_outputs[layer_name] = output.to(torch.float64).sum(dim=0)[0]
            if out_key in self.saved_outputs:
                return self.saved_outputs[out_key].permute(1, 0, 2).detach().clone()
            return output
        return hook

    def _rotate_weights(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    if module.weight.dtype == torch.int8:
                        weight_fp16 = module.weight.to(torch.float16)
                        rotated_weight = torch.rot90(weight_fp16, k=-1, dims=[0, 1])
                        module.weight = nn.Parameter(rotated_weight.to(dtype=torch.int8))
                    else:
                        module.weight = nn.Parameter(torch.rot90(module.weight, k=-1, dims=[0, 1]))

    def _restore_weights(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    if module.weight.dtype == torch.int8:
                        weight_fp16 = module.weight.to(torch.float16)
                        rotated_weight = torch.rot90(weight_fp16, k=1, dims=[0, 1])
                        module.weight = nn.Parameter(rotated_weight.to(dtype=torch.int8))
                    else:
                        module.weight = nn.Parameter(torch.rot90(module.weight, k=1, dims=[0, 1]))

    def register_hooks(self, start_value, steps):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                num_rows = module.weight.shape[1]
                self.hooks.append(module.register_forward_pre_hook(self._pre_hook(num_rows, start_value, steps)))
                self.hooks.append(module.register_forward_hook(self._forward_hook(name, num_rows)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_rotated_outSum_for_layer(self, layer_name, start_value=1, steps=30):
        module = self.model.get_submodule(layer_name)
        model_tmp = SimpleLinearModel(in_features=module.weight.shape[1], out_features=module.weight.shape[0])
        weight_fp16 = module.weight.to(torch.float16)
        rotated_weight = torch.rot90(weight_fp16, k=-1, dims=[0, 1])
        model_tmp.linear.weight = nn.Parameter(rotated_weight)
        input_tensor = torch.linspace(start_value, start_value + (steps - 1) * start_value, steps=steps, dtype=torch.float32, device=model_tmp.linear.weight.device).unsqueeze(0).unsqueeze(2).expand(1, steps, module.weight.shape[0])
        output = model_tmp(input_tensor)
        del model_tmp
        return output[0].to(torch.float64).sum(dim=0)

    def _run_for_fp8(self, device, start_value=1, steps=30):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.device != 'cuda' or module.weight.device != 'cpu':
                    continue
                self.rotated_outputs[name] = self.get_rotated_outSum_for_layer(name, start_value=start_value, steps=steps)
            gc.collect()
            torch.cuda.ipc_collect()

    def run(self, device, start_value=1, steps=30):
        if True:
            self._run_for_fp8(device, start_value=start_value, steps=steps)
        else:
            self._rotate_weights()
            self.steps = steps
            self.register_hooks(start_value, steps)
            input_ids = torch.arange(1, 1 + steps).unsqueeze(1).to(device)
            with torch.no_grad():
                self.model(input_ids=input_ids)
            self.remove_hooks()
            self._restore_weights()
        return (self.rotated_outputs, self.layer_hashes)