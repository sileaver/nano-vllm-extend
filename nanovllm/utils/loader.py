import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # Optional checkpoint-name handling (e.g. Qwen3.5 ships a multimodal
    # shell: "model.language_model.*" -> "model.*", visual/mtp skipped).
    remapping = getattr(model, "weight_remapping", ())
    ignored = getattr(model, "ignored_weight_prefixes", ())
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for ckpt_name in f.keys():
                if ckpt_name.startswith(ignored):
                    continue
                weight_name = ckpt_name
                for old, new in remapping:
                    if weight_name.startswith(old):
                        weight_name = new + weight_name[len(old):]
                        break
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(ckpt_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(ckpt_name))
