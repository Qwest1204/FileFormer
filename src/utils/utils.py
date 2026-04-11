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

import torch.nn.functional as F

def normalize_probabilities(logits, min_prob=1e-6, temperature=2.0):
    """
    Нормализует логиты в вероятности с контролем минимальной вероятности и температуры.

    Параметры:
    ----------
    logits : torch.Tensor
        Тензор сырых оценок формы (..., vocab_size).
    min_prob : float, optional (default=1e-5)
        Минимальная вероятность для каждого элемента после нормализации.
        Должно быть строго меньше 1 / vocab_size.
    temperature : float, optional (default=1.0)
        Температура softmax. Значения >1 делают распределение более равномерным,
        значения <1 — более острым. Используйте >1 для исправления «взрывной» вероятности.

    Возвращает:
    -----------
    torch.Tensor
        Тензор вероятностей той же формы, что и logits. Каждый элемент >= min_prob,
        сумма по последней оси равна 1, все значения < 1.
    """
    # Шаг 1: Softmax с температурой
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    vocab_size = probs.shape[-1]
    if min_prob * vocab_size >= 1.0:
        raise ValueError(f"min_prob ({min_prob}) слишком велико для vocab_size={vocab_size}. "
                         f"Необходимо min_prob < 1/{vocab_size}")

    # Шаг 2: Поднимаем все вероятности до min_prob, сохраняя сумму = 1
    # new_p = (1 - n*min_prob) * p + min_prob
    alpha = 1 - vocab_size * min_prob
    new_probs = alpha * probs + min_prob

    # Небольшая перенормировка для борьбы с погрешностями округления
    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)


    return new_probs

def convert_state_dict_to_lora(old_state_dict, lora_modules=['q_proj', 'v_proj']):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if any(f'.{mod}.' in key or key.endswith(f'.{mod}') for mod in lora_modules):
            if key.endswith('.weight'):
                new_key = key.replace('.weight', '.linear.weight')
            elif key.endswith('.bias'):
                new_key = key.replace('.bias', '.linear.bias')
            else:
                new_key = key
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def freeze_all_except_lora(model):
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False