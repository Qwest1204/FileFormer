import torch
import os
import yaml

def load_config(conf_path: str) -> dict:
    with open(conf_path, 'r') as f:
        data = yaml.safe_load(f)
    f.close()
    return data

def load_model(base_model, weights, device):
    base_model.load_state_dict(torch.load(weights, map_location=device))
    return base_model


