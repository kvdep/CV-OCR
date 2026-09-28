import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Добавление корня репозитория в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from src.dataset.dataset import Image2LatexDataset, make_collate_fn
from src.metrics.evaluator import EvaluationMetrics, calculate_ecdm
from src.models.transformer_ocr import FlexibleCNNTransformer
from src.utils.latex_render import render_latex_to_png, is_latex_available
from src.utils.system import (
    aggressive_memory_reset,
    compute_dynamic_batch_size,
    log_print,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка качества модели OCR LaTeX на тестовом сплите")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к файлу весов модели (.pt)")
    parser.add_argument("--base_path", type=str, default="./dataset/dataset_clean", help="Путь к очищенному датасету")
    parser.add_argument("--batch_size", type=int, default=None, help="Размер тестового батча")
    parser.add_argument("--beam_width", type=int, default=3, help="Ширина луча (Beam Width) при декодировании")
    parser.add_argument("--num_visual_samples", type=int, default=10, help="Количество примеров для рендеринга")
    parser.add_argument("--render_dir", type=str, default="./test_renders", help="Директория сохранения PNG рендеров")
    return parser.parse_args()


def evaluate(args):
    aggressive_memory_reset()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Файл чекпоинта не найден: {ckpt_path}")

    base_path = Path(args.base_path)
    vocab_path = base_path / "custom_t2i.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        t2i = json.load(f)
    i2t = {v: k for k, v in t2i.items()}
    vocab_size = len(t2i)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Загрузка чекпоинта: {ckpt_path} на устройство: {device}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    arch = checkpoint.get(
        "arch",
        {
            "vocab_size": vocab_size,
            "d_model": 256,
            "nhead": 8,
            "num_decoder_layers": 8,
            "max_len": 250,
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
    model.eval()

    computed_batch_size, _ = compute_dynamic_batch_size()
    batch_size = args.batch_size if args.batch_size is not None else computed_batch_size

    test_img_folder = base_path / "images"
    test_filenames = base_path / "data" / "test" / "images_test.txt"
    test_formulas = base_path / "data" / "test" / "formulas_test.txt"

    test_dataset = Image2LatexDataset(
        str(test_img_folder),
        str(test_filenames),
        str(test_formulas),
        t2i,
    )
    collate_fn = make_collate_fn(t2i["<P>"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=t2i["<P>"])
    evaluator = EvaluationMetrics()

    total_samples = len(test_dataset)
    log_print(
        f"Оценка на {total_samples} сэмплах "
        f"(Batch Size = {batch_size}, Beam Width = {args.beam_width})..."
    )

    all_predictions = []
    all_targets = []
    start_time = time.time()

    with torch.no_grad():
        for step_idx, (imgs, ids, raw_fms) in enumerate(test_loader):
            imgs = imgs.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            imgs_norm = TF.normalize(imgs.float() / 255.0, mean=[0.0672], std=[0.1625])

            outputs = model(imgs_norm, ids[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
            loss_val = loss.item() * imgs.size(0)

            preds = model.predict(
                imgs_norm,
                t2i["<S>"],
                t2i["<E>"],
                max_len=arch.get("max_len", 250),
                beam_width=args.beam_width,
            )
            preds_list = preds.cpu().tolist()

            for p_tokens, raw_target in zip(preds_list, raw_fms):
                pred_syms = [
                    i2t[tok]
                    for tok in p_tokens
                    if tok not in {t2i["<S>"], t2i["<E>"], t2i["<P>"]}
                ]
                pred_str = " ".join(pred_syms)
                target_str = raw_target.strip()

                evaluator.update(pred_str, target_str, loss_val=loss.item())
                all_predictions.append(pred_str)
                all_targets.append(target_str)

            if (step_idx + 1) % 50 == 0 or (step_idx + 1) == len(test_loader):
                processed = min((step_idx + 1) * batch_size, total_samples)
                cur_summary = evaluator.compute_summary()
                log_print(
                    f"Обработано: {processed}/{total_samples} сэмплов | "
                    f"Промежуточный Macro-ECDM (Char): {cur_summary['macro_char_ecdm']:.2f}%"
                )

    elapsed = time.time() - start_time
    summary = evaluator.compute_summary()

    log_print(f"Тестирование завершено за {elapsed:.1f} с.")
    print("=" * 60)
    print(f"Test Loss:                  {summary['mean_loss']:.4f}")
    print(f"Exact Match Rate:           {summary['exact_match_rate']:.2f}%")
    print(f"Macro-Averaged ECDM (Char): {summary['macro_char_ecdm']:.2f}%")
    print(f"Macro-Averaged ECDM (Token):{summary['macro_token_ecdm']:.2f}%")
    print(f"Micro-Averaged CER:         {summary['micro_cer']:.2f}%")
    print(f"Micro-Averaged TER:         {summary['micro_ter']:.2f}%")
    print("=" * 60)

    # Рендеринг визуальных сэмплов
    if args.num_visual_samples > 0 and is_latex_available():
        render_dir = Path(args.render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)
        visual_indices = random.sample(
            range(total_samples), min(args.num_visual_samples, total_samples)
        )
        for rank, idx in enumerate(visual_indices):
            p_str = all_predictions[idx]
            t_str = all_targets[idx]
            sample_ecdm = calculate_ecdm(p_str, t_str) * 100.0
            print(
                f"Sample {rank + 1:02d} | Exact: {p_str == t_str} | "
                f"ECDM: {sample_ecdm:.1f}% | Target: {t_str} | Pred: {p_str}"
            )
            render_path = render_dir / f"sample_{rank + 1}.png"
            render_latex_to_png(p_str, output_filename=str(render_path))

    # Сводная таблица сравнения с бейзлайном
    comp_df = pd.DataFrame(
        {
            "Модель": [
                "FlexibleCNNTransformer Baseline",
                f"FlexibleCNNTransformer ({ckpt_path.stem})",
            ],
            "Test Loss": [0.6514, summary["mean_loss"]],
            "Exact Match (%)": [12.81, summary["exact_match_rate"]],
            "Macro ECDM (%)": [79.29, summary["macro_char_ecdm"]],
            "Micro CER (%)": [28.63, summary["micro_cer"]],
            "Micro TER (%)": [None, summary["micro_ter"]],
        }
    )
    print("\nСводная таблица результатов:")
    print(comp_df.to_string(index=False))

    aggressive_memory_reset()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
