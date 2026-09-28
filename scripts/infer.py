import argparse
import json
import sys
from pathlib import Path

# Добавление корня репозитория в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from src.models.transformer_ocr import FlexibleCNNTransformer
from src.utils.latex_render import render_latex_to_png, is_latex_available


def parse_args():
    parser = argparse.ArgumentParser(description="Распознавание формулы LaTeX с одиночного изображения")
    parser.add_argument("--image", type=str, required=True, help="Путь к файлу входного изображения (.png, .jpg)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к весам обученной модели (.pt)")
    parser.add_argument("--vocab", type=str, default="./dataset/dataset_clean/custom_t2i.json", help="Путь к словарю custom_t2i.json")
    parser.add_argument("--beam_width", type=int, default=3, help="Ширина луча (Beam Width) при поиске")
    parser.add_argument("--max_len", type=int, default=250, help="Максимальное число токенов генерации")
    parser.add_argument("--render", type=str, default=None, help="Путь для сохранения скомпилированного PNG (опционально)")
    return parser.parse_args()


def preprocess_image(image_path: str, target_size=(1024, 128)) -> torch.Tensor:
    """Загрузка, инверсия и нормализация одиночного изображения под формат модели."""
    img = Image.open(image_path).convert("L")
    if np.mean(np.array(img)) > 127:
        img = ImageOps.invert(img)

    w, h = img.size
    target_w, target_h = target_size
    ratio = min(target_w / max(w, 1), target_h / max(h, 1))
    nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
    img = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("L", (target_w, target_h), 0)
    canvas.paste(img, (0, 0))

    tensor = torch.from_numpy(np.array(canvas, dtype=np.uint8)).unsqueeze(0).unsqueeze(0)
    tensor_norm = TF.normalize(tensor.float() / 255.0, mean=[0.0672], std=[0.1625])
    return tensor_norm


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {img_path}")

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Файл словаря не найден: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        t2i = json.load(f)
    i2t = {v: k for k, v in t2i.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = checkpoint.get("arch", {
        "vocab_size": len(t2i),
        "d_model": 256,
        "nhead": 8,
        "num_decoder_layers": 8,
        "max_len": args.max_len,
    })

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

    img_tensor = preprocess_image(str(img_path)).to(device)

    with torch.no_grad():
        preds = model.predict(
            img_tensor,
            t2i["<S>"],
            t2i["<E>"],
            max_len=args.max_len,
            beam_width=args.beam_width,
        )

    tokens = preds[0].cpu().tolist()
    pred_symbols = [
        i2t[tok] for tok in tokens if tok not in {t2i["<S>"], t2i["<E>"], t2i["<P>"]}
    ]
    latex_result = " ".join(pred_symbols)

    print("=" * 60)
    print("РАСПОЗНАННАЯ ФОРМУЛА LATEX:")
    print(latex_result)
    print("=" * 60)

    if args.render:
        if is_latex_available():
            ok = render_latex_to_png(latex_result, output_filename=args.render)
            if ok:
                print(f"Скомпилированное изображение сохранено в: {args.render}")
            else:
                print(f"Ошибка компиляции формулы через LaTeX в {args.render}")
        else:
            print("Предупреждение: утилиты latex/dvipng не найдены в PATH. Рендеринг пропущен.")


if __name__ == "__main__":
    main()
