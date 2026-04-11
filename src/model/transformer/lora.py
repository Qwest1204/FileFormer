import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=0, lora_alpha=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank if rank > 0 else 1.0

        # Обычный линейный слой (его веса будут заморожены для LoRA-обучения)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Замораживаем веса обычного слоя
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA параметры (если rank > 0)
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            # Инициализация: A - случайно (часто Kaiming), B - нули
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # стандартная инициализация для LoRA
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        # Флаг для включения/выключения LoRA на лету (удобно для инференса)
        self.lora_enabled = rank > 0  # по умолчанию включено, если rank>0

    def enable_lora(self, enabled=True):
        """Позволяет временно отключить LoRA (например, для сравнения)"""
        if self.rank > 0:
            self.lora_enabled = enabled

    def forward(self, x):
        # Базовый вывод через замороженный линейный слой
        out = self.linear(x)

        # Добавляем LoRA-вклад, если включено и rank>0
        if self.lora_enabled and self.rank > 0:
            # x: (batch, ..., in_features)
            # lora_A: (rank, in_features) -> (x @ lora_A.T) даст (..., rank)
            # lora_B: (out_features, rank) -> ((x @ lora_A.T) @ lora_B.T) даст (..., out_features)
            out = out + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return out

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, lora_enabled={self.lora_enabled}'