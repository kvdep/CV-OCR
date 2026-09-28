import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Tuple


def setup_kaggle() -> Tuple[str, str]:
    """Настройка учетных данных Kaggle из переменных окружения или файла ~/.kaggle/kaggle.json.
    
    Возвращает кортеж (kaggle_user, kaggle_key).
    """
    kaggle_user = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_config_dir / "kaggle.json"

    if not kaggle_user or not kaggle_key:
        if kaggle_json_path.exists():
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                kaggle_user = data.get("username")
                kaggle_key = data.get("key")
                os.environ["KAGGLE_USERNAME"] = kaggle_user
                os.environ["KAGGLE_KEY"] = kaggle_key
        else:
            raise ValueError(
                "Учетные данные Kaggle не найдены. "
                "Задайте переменные KAGGLE_USERNAME и KAGGLE_KEY в терминале "
                f"или создайте файл {kaggle_json_path}"
            )
    else:
        kaggle_config_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json_path, "w", encoding="utf-8") as f:
            json.dump({"username": kaggle_user, "key": kaggle_key}, f)
        if os.name != "nt":
            os.chmod(kaggle_json_path, 0o600)

    return kaggle_user, kaggle_key


def extract_zip(archive_path: str, target_dir: str) -> None:
    """Кроссплатформенная распаковка zip-архива средствами Python."""
    arch = Path(archive_path)
    dst = Path(target_dir)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arch, "r") as zf:
        zf.extractall(dst)


def create_zip(source_dir: str, target_zip_path: str) -> None:
    """Кроссплатформенное сжатие директории в zip-архив."""
    src = Path(source_dir)
    dst = Path(target_zip_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in src.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(src))
