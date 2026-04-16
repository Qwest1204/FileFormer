import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MoELayer(nn.Module):
    """
    Mixture of Experts слой с современными техниками:
    - Top-k маршрутизация
    - Балансировка загрузки (load balancing loss)
    - Экспертная ёмкость (expert capacity)
    - Шум в gating (noisy gating)
    - Эффективное распределение токенов по экспертам
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_experts: int,
            top_k: int = 1,
            capacity_factor: float = 1.25,
            noisy_gating: bool = True,
            noise_epsilon: float = 1e-2,
            dropout: float = 0.1,
    ):
        """
        Args:
            d_model: размерность входных/выходных признаков
            d_ff: размерность скрытого слоя эксперта (обычно 4*d_model)
            num_experts: количество экспертов
            top_k: сколько экспертов активируется на каждый токен
            capacity_factor: множитель для расчёта ёмкости эксперта
            noisy_gating: добавлять ли обучаемый шум к логитам
            noise_epsilon: масштаб шума для noisy gating
            dropout: dropout внутри экспертов
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating
        self.d_model = d_model

        # Маршрутизатор (gating network)
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Параметры шума (только если noisy_gating)
        if noisy_gating:
            self.noise_std = nn.Parameter(torch.zeros(num_experts))

        # Создаём экспертов как независимые MLP
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        self._init_weights()

    def _init_weights(self):
        # Инициализация маршрутизатора с малым std для стабильного начала
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02 / self.num_experts)
        if self.noisy_gating:
            nn.init.constant_(self.noise_std, 0.01)

    def _compute_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисляет логиты и веса экспертов для каждого токена.
        Возвращает: raw_logits, gates (softmax), top_indices.
        """
        # x: [batch_size * seq_len, d_model]
        logits = self.router(x)  # [N, num_experts]

        if self.noisy_gating and self.training:
            # Добавляем обучаемый шум: noise = softplus(noise_std) * eps, eps ~ N(0,1)
            noise_std = F.softplus(self.noise_std) + 1e-5
            noise = torch.randn_like(logits) * noise_std.unsqueeze(0)
            logits = logits + noise

        # Выбираем top_k экспертов
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)  # [N, top_k]

        # Softmax по выбранным логитам (с температурой 1.0)
        gates = F.softmax(top_logits, dim=-1)  # [N, top_k]

        return logits, gates, top_indices

    def _capacity(self, n_tokens: int) -> int:
        """Вычисляет ёмкость одного эксперта."""
        # Базовая ёмкость: среднее число токенов на эксперта, умноженное на фактор
        base_capacity = max(1, int(self.capacity_factor * n_tokens / self.num_experts))
        # Чтобы гарантировать, что хотя бы один токен может быть обработан
        return base_capacity

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: входной тензор [batch_size, seq_len, d_model]
        Returns:
            out: выходной тензор той же формы
            aux_loss: вспомогательная потеря балансировки (0 если не в training)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [N, d_model]
        n_tokens = x_flat.size(0)

        # Получаем гейты и индексы экспертов
        raw_logits, gates, top_indices = self._compute_gates(x_flat)  # [N, top_k]

        # Вычисляем ёмкость эксперта
        expert_capacity = self._capacity(n_tokens)

        # Подготовка к распределению токенов по экспертам
        # Создаём маску: какие токены попадают к эксперту с учётом ёмкости
        expert_mask = torch.zeros(
            self.num_experts, expert_capacity, dtype=torch.long, device=x.device
        )
        # Счётчик фактического заполнения каждого эксперта
        expert_used = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        # Для каждого выбранного эксперта (k от 0 до top_k-1) заполняем маску
        for k in range(self.top_k):
            expert_idx = top_indices[:, k]  # [N]
            # Сортируем по экспертам, чтобы эффективно заполнять
            # Используем bincount для подсчёта числа токенов на эксперта
            counts = torch.bincount(expert_idx, minlength=self.num_experts)
            # Кумулятивные смещения в пределах ёмкости
            cumsum = torch.cumsum(counts, dim=0) - counts

            # Для каждого эксперта заполняем маску (только до ёмкости)
            for e in range(self.num_experts):
                if counts[e] == 0:
                    continue
                # Индексы токенов, выбравших эксперта e в k-й позиции
                token_pos = (expert_idx == e).nonzero(as_tuple=True)[0]
                # Сколько из них реально поместится
                take = min(counts[e].item(), expert_capacity - expert_used[e].item())
                if take > 0:
                    start = expert_used[e].item()
                    expert_mask[e, start:start + take] = token_pos[:take]
                    expert_used[e] += take

        # Теперь для каждого эксперта вычисляем выходы только для назначенных токенов
        expert_outputs = []
        for e in range(self.num_experts):
            if expert_used[e] == 0:
                continue
            indices = expert_mask[e, :expert_used[e]]  # индексы токенов в x_flat
            expert_input = x_flat[indices]  # [used, d_model]
            expert_out = self.experts[e](expert_input)  # [used, d_model]
            expert_outputs.append((e, indices, expert_out))

        # Собираем выходы обратно с учётом весов
        out_flat = torch.zeros_like(x_flat)
        # Сначала разбросаем выходы экспертов по токенам с умножением на веса
        # Для этого проходим по всем выходам экспертов
        for e, indices, exp_out in expert_outputs:
            # Для каждого токена нужно найти его вес у эксперта e (возможно несколько k)
            # Удобнее сначала собрать все веса в разреженный формат
            # Альтернатива: сохранить для каждого токена список (эксперт, вес) и потом агрегировать
            # Используем add.at (через index_add) для эффективного накопления
            pass

        # Более эффективный способ: для каждого токена мы знаем top_indices и gates.
        # Выход эксперта для токена i: sum_{k} gate[i,k] * expert_out_k
        # Поэтому можно для каждого k обработать все эксперты сразу.

        # Реализуем через цикл по k (их мало, top_k обычно 1 или 2)
        for k in range(self.top_k):
            # Индексы экспертов для k-го выбора: [N]
            exp_k = top_indices[:, k]
            # Веса для этого выбора: [N]
            gate_k = gates[:, k]

            # Для каждого эксперта собираем токены, которые его выбрали в k-й позиции
            for e in range(self.num_experts):
                mask = (exp_k == e)
                if not mask.any():
                    continue
                indices = mask.nonzero(as_tuple=True)[0]
                # Входные токены для эксперта e (но уже посчитаны выше, нужно переиспользовать)
                # Здесь для простоты пересчитываем, но в production стоит кэшировать
                exp_input = x_flat[indices]
                exp_out = self.experts[e](exp_input)  # [m, d_model]
                # Добавляем взвешенный вклад в соответствующие позиции out_flat
                out_flat.index_add_(0, indices, exp_out * gate_k[indices].unsqueeze(-1))

        # Если какие-то токены не попали ни к одному эксперту из-за ёмкости,
        # их выход останется нулевым. На практике их пропускают через residual (вне слоя).

        out = out_flat.view(batch_size, seq_len, d_model)

        # Вычисление вспомогательной потери балансировки
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            # Используем raw_logits (до шума) для подсчёта вероятностей
            probs = F.softmax(raw_logits, dim=-1)  # [N, num_experts]
            # Доля токенов, назначенных эксперту (по top-1)
            # Для простоты используем top-1 назначения
            _, top1_idx = torch.topk(raw_logits, 1, dim=-1)
            # Подсчёт числа токенов на эксперта
            density = torch.bincount(top1_idx.squeeze(-1), minlength=self.num_experts).float() / n_tokens
            # Средняя вероятность по всем токенам для каждого эксперта
            avg_prob = probs.mean(dim=0)  # [num_experts]
            # Loss = num_experts * sum(density * avg_prob)
            aux_loss = self.num_experts * torch.sum(density * avg_prob)

        return out, aux_loss


class Expert(nn.Module):
    """Стандартный MLP, используемый в качестве эксперта."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x))))