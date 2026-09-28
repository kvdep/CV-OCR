import argparse
import sys
from pathlib import Path

# Добавление корня репозитория в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.preprocessing import fast_build_clean_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка, очистка и стандартизация датасета формул Im2Latex"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./dataset/dataset",
        help="Путь к исходным сырым сплитам и изображениям",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./dataset/dataset_clean",
        help="Путь для сохранения очищенного датасета и custom_t2i.json",
    )
    args = parser.parse_args()

    print(f"Запуск предобработки: {args.source_dir} -> {args.target_dir}")
    t2i = fast_build_clean_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
    )
    print(f"Готово. Сформирован словарь из {len(t2i)} токенов.")


if __name__ == "__main__":
    main()
