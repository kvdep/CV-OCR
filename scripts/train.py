import argparse
import json
import os
import sys
from pathlib import Path

# Добавление корня репозитория в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from configs.default_config import ModelConfig, TrainingConfig, DataConfig
from src.dataset.dataset import Image2LatexDataset, make_collate_fn
from src.dataset.samplers import GroupedBatchSampler
from src.models.transformer_ocr import FlexibleCNNTransformer
from src.utils.system import (
    get_memory_info,
    aggressive_memory_reset,
    compute_dynamic_batch_size,
    log_print,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели FlexibleCNNTransformer для OCR LaTeX")
    parser.add_argument("--model_name", type=str, default="im2latex_model", help="Имя модели и сохраняемого файла весов")
    parser.add_argument("--base_path", type=str, default="./dataset/dataset_clean", help="Путь к очищенному датасету")
    parser.add_argument("--num_epochs", type=int, default=150, help="Максимальное количество эпох")
    parser.add_argument("--batch_size", type=int, default=None, help="Размер батча (None для динамического расчета)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Начальная скорость обучения (learning rate)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Коэффициент weight decay для AdamW")
    parser.add_argument("--patience", type=int, default=15, help="Терпение Early Stopping (эпох без улучшения)")
    parser.add_argument("--num_workers", type=int, default=None, help="Число параллельных воркеров DataLoader")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Коэффициент сглаживания меток (Label Smoothing)")
    parser.add_argument("--weight_ratio", type=float, default=5.0, help="Максимальное отношение весов редких классов к частым")
    parser.add_argument("--beam_width", type=int, default=3, help="Ширина луча для валидационной генерации")
    parser.add_argument("--d_model", type=int, default=256, help="Размерность скрытого представления")
    parser.add_argument("--nhead", type=int, default=8, help="Количество голов внимания в декодере")
    parser.add_argument("--num_decoder_layers", type=int, default=8, help="Количество слоев Transformer Decoder")
    parser.add_argument("--max_len", type=int, default=250, help="Максимальная длина генерируемой последовательности")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Директория сохранения чекпоинтов")
    return parser.parse_args()


def train(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    aggressive_memory_reset()

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    computed_batch_size, computed_workers = compute_dynamic_batch_size()
    batch_size = args.batch_size if args.batch_size is not None else computed_batch_size
    num_workers = args.num_workers if args.num_workers is not None else computed_workers

    log_print(
        f"Параметры запуска: Batch Size = {batch_size}, "
        f"Num Workers = {num_workers}, Модель = {args.model_name}"
    )

    base_path = Path(args.base_path)
    vocab_path = base_path / "custom_t2i.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Файл словаря не найден: {vocab_path}. "
            "Сначала выполните: python scripts/prepare_data.py"
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        t2i = json.load(f)
    i2t = {v: k for k, v in t2i.items()}
    vocab_size = len(t2i)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = device.type == "cuda"

    checkpoint_path = ckpt_dir / f"{args.model_name}.pt"
    best_checkpoint_path = ckpt_dir / f"best_{args.model_name}.pt"

    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    es_counter = 0

    if checkpoint_path.exists():
        log_print(f"Загрузка существующего чекпоинта: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        arch = checkpoint.get(
            "arch",
            {
                "vocab_size": vocab_size,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_decoder_layers": args.num_decoder_layers,
                "max_len": args.max_len,
            },
        )
        model = FlexibleCNNTransformer(
            vocab_size=arch["vocab_size"],
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            num_decoder_layers=arch["num_decoder_layers"],
            max_len=arch["max_len"],
        ).to(device)
        model.t2i = t2i
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", history)
        es_counter = checkpoint.get("es_counter", 0)
        log_print(f"Возобновление обучения с эпохи {start_epoch}")
    else:
        log_print(f"Инициализация новой модели {args.model_name} (vocab_size={vocab_size})")
        model = FlexibleCNNTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_decoder_layers=args.num_decoder_layers,
            max_len=args.max_len,
        ).to(device)
        model.t2i = t2i
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    scaler = torch.amp.GradScaler("cuda") if use_pin else None

    # Загрузка обучающей и валидационной выборок
    train_img_folder = base_path / "images"
    train_filenames = base_path / "data" / "train" / "images_train.txt"
    train_formulas = base_path / "data" / "train" / "formulas_train.txt"
    val_filenames = base_path / "data" / "val" / "images_val.txt"
    val_formulas = base_path / "data" / "val" / "formulas_val.txt"

    train_dataset = Image2LatexDataset(str(train_img_folder), str(train_filenames), str(train_formulas), t2i)
    val_dataset = Image2LatexDataset(str(train_img_folder), str(val_filenames), str(val_formulas), t2i)

    # Расчет обратных частотных весов классов
    counts = np.ones(vocab_size, dtype=np.float32)
    for _, ids, _ in train_dataset.samples:
        for idx in ids:
            counts[idx] += 1.0
    freqs = counts / counts.sum()
    weights = 1.0 / (freqs + 1e-5)
    weights /= weights.min()
    weights = np.clip(weights, 1.0, args.weight_ratio if args.weight_ratio is not None else 1.0)
    weights[t2i["<P>"]] = 0.0
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=t2i["<P>"],
        label_smoothing=args.label_smoothing,
        reduction="none",
    )

    collate_fn = make_collate_fn(t2i["<P>"])

    mem_info = get_memory_info()
    usable_vram = max(0.5, mem_info["free_vram_gb"] - 1.0)
    max_tokens_dynamic = max(1000, min(int(usable_vram * 400), 16000))

    train_sampler = GroupedBatchSampler(train_dataset, batch_size, shuffle=True, max_tokens=max_tokens_dynamic)
    val_sampler = GroupedBatchSampler(val_dataset, batch_size, shuffle=False, max_tokens=max_tokens_dynamic)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    def compute_weighted_loss(pred_logits, target_tokens):
        raw_loss = criterion(pred_logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
        tok_weights = weights_tensor[target_tokens.reshape(-1)]
        return (raw_loss * tok_weights).sum() / (tok_weights.sum() + 1e-8)

    try:
        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            total_steps = len(train_loader)
            train_loss = 0.0

            for step_idx, (imgs, ids, _) in enumerate(train_loader):
                imgs = imgs.to(device, non_blocking=True)
                ids = ids.to(device, non_blocking=True)
                imgs = TF.normalize(imgs.float() / 255.0, mean=[0.0672], std=[0.1625])
                optimizer.zero_grad(set_to_none=True)

                try:
                    if scaler is not None:
                        with torch.amp.autocast("cuda"):
                            outputs = model(imgs, ids[:, :-1])
                            loss = compute_weighted_loss(outputs, ids[:, 1:])
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(imgs, ids[:, :-1])
                        loss = compute_weighted_loss(outputs, ids[:, 1:])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    train_loss += loss.item()
                    step_loss_val = loss.item()
                except torch.cuda.OutOfMemoryError:
                    # Восстановление при нехватке VRAM через разделение на микробатчи
                    optimizer.zero_grad(set_to_none=True)
                    aggressive_memory_reset()
                    micro_batch_size = max(2, imgs.size(0) // 4)
                    b_sz = imgs.size(0)
                    step_loss_val = 0.0
                    num_micros = (b_sz + micro_batch_size - 1) // micro_batch_size

                    for m_i in range(0, b_sz, micro_batch_size):
                        sub_imgs = imgs[m_i : m_i + micro_batch_size]
                        sub_ids = ids[m_i : m_i + micro_batch_size]
                        if scaler is not None:
                            with torch.amp.autocast("cuda"):
                                sub_out = model(sub_imgs, sub_ids[:, :-1])
                                sub_loss = compute_weighted_loss(sub_out, sub_ids[:, 1:]) / num_micros
                            scaler.scale(sub_loss).backward()
                        else:
                            sub_out = model(sub_imgs, sub_ids[:, :-1])
                            sub_loss = compute_weighted_loss(sub_out, sub_ids[:, 1:]) / num_micros
                            sub_loss.backward()
                        step_loss_val += sub_loss.item() * num_micros

                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    train_loss += step_loss_val

                if step_idx % 20 == 0 or step_idx == total_steps - 1:
                    log_print(
                        f"Эпоха {epoch:03d} | Шаг {step_idx}/{total_steps} | "
                        f"Потери батча: {step_loss_val:.4f}"
                    )

            train_loss /= total_steps

            # Валидационный проход
            model.eval()
            total_val_steps = len(val_loader)
            val_loss = 0.0
            val_preds, val_targets, val_raw_formulas = [], [], []

            with torch.no_grad():
                for step_idx, (imgs, ids, raw_fms) in enumerate(val_loader):
                    imgs = imgs.to(device, non_blocking=True)
                    ids = ids.to(device, non_blocking=True)
                    imgs_norm = TF.normalize(imgs.float() / 255.0, mean=[0.0672], std=[0.1625])
                    outputs = model(imgs_norm, ids[:, :-1])
                    loss = compute_weighted_loss(outputs, ids[:, 1:])
                    val_loss += loss.item()

                    if step_idx == 0:
                        preds = model.predict(
                            imgs_norm[:5],
                            t2i["<S>"],
                            t2i["<E>"],
                            max_len=args.max_len,
                            beam_width=args.beam_width,
                        )
                        val_preds.extend(preds.cpu().tolist())
                        val_targets.extend(ids[:5].cpu().tolist())
                        val_raw_formulas.extend(raw_fms[:5])

            val_loss /= total_val_steps
            log_print(f"Эпоха {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            for rank, (p_tokens, _, raw_fm) in enumerate(
                zip(val_preds[:3], val_targets[:3], val_raw_formulas[:3])
            ):
                pred_syms = [
                    i2t[tok]
                    for tok in p_tokens
                    if tok not in {t2i["<S>"], t2i["<E>"], t2i["<P>"]}
                ]
                log_print(f"Val Sample {rank + 1} | Target: {raw_fm} | Pred: {' '.join(pred_syms)}")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            checkpoint_data = {
                "epoch": epoch,
                "arch": model.get_architecture_config(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
                "es_counter": es_counter,
            }
            torch.save(checkpoint_data, checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_data["best_val_loss"] = best_val_loss
                torch.save(checkpoint_data, best_checkpoint_path)
                log_print(f"Обновлен лучший чекпоинт: {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")
                es_counter = 0
            else:
                es_counter += 1

            scheduler.step(val_loss)
            if es_counter >= args.patience:
                log_print(f"Сработало раннее прерывание (Early Stopping) на эпохе {epoch}")
                break

    except KeyboardInterrupt:
        log_print("Обучение прервано пользователем. Сохранение текущего состояния...")
        checkpoint_data = {
            "epoch": epoch if "epoch" in locals() else start_epoch,
            "arch": model.get_architecture_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "es_counter": es_counter,
        }
        torch.save(checkpoint_data, checkpoint_path)
    finally:
        aggressive_memory_reset()


if __name__ == "__main__":
    args = parse_args()
    train(args)
