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