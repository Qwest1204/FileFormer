import torch
import os
import yaml

def load_config(conf_path: str) -> dict:
    with open(conf_path, 'r') as f:
        data = yaml.safe_load(f)
    f.close()
    return data

def save_model(model, optimizer, config: dict, model_name: str) -> None:
    try:
        torch.save({"model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "config": config}, model_name)
        print("successfully saved model")
    except Exception as e:
        print(e)


