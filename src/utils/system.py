import gc
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict

import torch

try:
    import psutil
except ImportError:
    psutil = None


_last_print_time = time.time()


def get_memory_info() -> Dict[str, float]:
    """Возвращает информацию о доступной оперативной и видеопамяти в гигабайтах.
    
    Работает кроссплатформенно на Windows и Linux.
    """
    if psutil is not None:
        try:
            free_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            free_ram_gb, total_ram_gb = 4.0, 8.0
    else:
        # Fallback при отсутствии psutil
        free_ram_gb, total_ram_gb = 4.0, 8.0

    free_vram_gb, total_vram_gb = 0.0, 0.0
    if torch.cuda.is_available():
        try:
            free_vram, total_vram = torch.cuda.mem_get_info()
            free_vram_gb = free_vram / (1024 ** 3)
            total_vram_gb = total_vram / (1024 ** 3)
        except Exception:
            free_vram_gb, total_vram_gb = 0.0, 0.0

    return {
        "free_ram_gb": free_ram_gb,
        "total_ram_gb": total_ram_gb,
        "free_vram_gb": free_vram_gb,
        "total_vram_gb": total_vram_gb,
    }


def aggressive_memory_reset(min_free_vram_gb: float = 0.5) -> None:
    """Принудительно очищает неиспользуемую память Python и кэши CUDA.
    
    Вызывает gc.collect(), torch.cuda.empty_cache() и сбрасывает статистику пикового потребления.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        mem_info = get_memory_info()
        if mem_info["free_vram_gb"] < min_free_vram_gb:
            raise RuntimeError(
                f"Критическая нехватка VRAM: доступно {mem_info['free_vram_gb']:.2f} ГБ "
                f"(минимум: {min_free_vram_gb:.2f} ГБ). Требуется завершить фоновые процессы GPU."
            )


def compute_dynamic_batch_size(
    min_batch: int = 8,
    max_batch: int = 48,
    vram_margin_gb: float = 1.0,
) -> Tuple[int, int]:
    """Рассчитывает размер батча и количество воркеров на основе доступных ресурсов системы."""
    mem_info = get_memory_info()
    free_vram_gb = mem_info["free_vram_gb"]
    free_ram_gb = mem_info["free_ram_gb"]

    if torch.cuda.is_available():
        usable_vram_gb = max(0.5, free_vram_gb - vram_margin_gb)
        raw_batch = int(usable_vram_gb * 2.0)
        computed_batch_size = max(min_batch, min((raw_batch // 8) * 8, max_batch))
    else:
        computed_batch_size = min_batch

    # На Windows при num_workers > 0 процесс спавнит дочерние интерпретаторы
    computed_num_workers = 2 if free_ram_gb > 2.0 else 0
    return computed_batch_size, computed_num_workers


def log_print(*args, **kwargs) -> None:
    """Форматированный вывод в консоль с таймстемпом, дельтой времени, типом устройства и памятью."""
    global _last_print_time
    now = datetime.now(timezone(timedelta(hours=3)))
    elapsed = time.time() - _last_print_time
    _last_print_time = time.time()

    eh = int(elapsed // 3600)
    em = int((elapsed % 3600) // 60)
    es = int(elapsed % 60)
    elapsed_str = f"{eh:02d}:{em:02d}:{es:02d}"

    dev = "GPU" if torch.cuda.is_available() else "CPU"
    mem_info = get_memory_info()
    ram_str = f"[{mem_info['free_ram_gb']:.1f}]"
    vram_str = f" [{mem_info['free_vram_gb']:.1f}]" if dev == "GPU" else ""

    msc_time_str = now.strftime("%H:%M:%S")
    prefix = f"[{msc_time_str}] [{elapsed_str}] [{dev}] {ram_str}{vram_str}"
    text = " ".join(map(str, args))
    print(f"{prefix} {text}", **kwargs)
    sys.stdout.flush()
