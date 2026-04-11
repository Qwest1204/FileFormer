import torch
import time
from typing import List, Dict, Any

# Предполагается, что класс FileFormer определён в модуле model
from model import FileFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_forward_speed(
    model: Any,
    batch_size: int = 1,
    seq_lengths: List[int] = [512, 1024, 2048, 8196, 16256, 32512],
    dtypes: List[str] = ["fp32", "fp16", "fp8"],
    warmup_iters: int = 5,
    measure_iters: int = 10,
    vocab_size: int = 10000,
) -> Dict[int, Dict[str, float]]:
    """
    Тест скорости forward с autocast для fp16/fp8.
    Модель всегда в fp32, внутри контекста выполняются операции в нужной точности.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.eval()

    results = {}

    for seq_len in seq_lengths:
        results[seq_len] = {}
        print(f"\n--- Sequence length: {seq_len} ---")

        for dtype in dtypes:
            print(f"  Precision: {dtype}")

            # Выбор контекста автокаста
            if dtype == "fp32":
                autocast_context = torch.no_grad()  # нет автокаста
            elif dtype == "fp16":
                autocast_context = torch.amp.autocast(device_type=device.type, dtype=torch.float16)
            elif dtype == "fp8":
                if not hasattr(torch, "fp8_autocast") or device.type != "cuda":
                    print("    FP8 not supported (requires torch.fp8_autocast and CUDA), skipping")
                    continue
                autocast_context = torch.fp8_autocast(enabled=True, dtype=torch.float8_e4m3fn)
            else:
                print(f"    Unknown dtype {dtype}, skipping")
                continue

            # Генерация входных токенов
            try:
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
            except RuntimeError as e:
                print(f"    Failed to allocate input: {e}")
                continue

            # Прогрев
            warmup_ok = True
            with torch.no_grad():
                for _ in range(warmup_iters):
                    try:
                        with autocast_context:
                            _ = model(input_ids)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("    OOM during warmup, skipping")
                            torch.cuda.empty_cache()
                            warmup_ok = False
                            break
                        else:
                            raise
            if not warmup_ok:
                continue

            # Измерение
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()

            for _ in range(measure_iters):
                with autocast_context:
                    _ = model(input_ids)

            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = (time.time() - start_time) / measure_iters
            results[seq_len][dtype] = elapsed
            print(f"    Average forward time: {elapsed * 1000:.2f} ms")

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results



# Пример использования
if __name__ == "__main__":
    model = FileFormer(
        vocab_size=261,
        embed_size=256,
        n_heads=4,
        n_layers=6,
        drop_rate=0.1

    ).to(device)

    speed_results = test_forward_speed(
        model=model,
        batch_size=1,
        seq_lengths=[512, 1024, 2048, 8196, 16256, 32512],
        dtypes=["fp32", "fp16", "fp8"],
        warmup_iters=5,
        measure_iters=10,
        vocab_size=256,
    )

    # Вывод результатов в виде таблицы
    print("\n" + "=" * 60)
    print("RESULTS (average forward time in milliseconds)")
    print("SeqLen\\Prec", end="")
    for dtype in ["fp32", "fp16", "fp8"]:
        print(f"\t{dtype}", end="")
    print()
    for seq_len, times in speed_results.items():
        print(f"{seq_len}", end="")
        for dtype in ["fp32", "fp16", "fp8"]:
            t = times.get(dtype)
            if t is not None:
                print(f"\t{t * 1000:.2f}", end="")
            else:
                print(f"\t---", end="")
        print()