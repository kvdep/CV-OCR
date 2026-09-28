import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def is_latex_available() -> bool:
    """Проверяет наличие в системе исполняемых бинарных файлов latex и dvipng."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def render_latex_to_png(
    formula_string: str,
    output_filename: str = "render.png",
    dpi: int = 200,
    temp_dir: Optional[str] = None,
) -> bool:
    """Компилирует строку формулы LaTeX в изображение PNG с помощью standalone, latex и dvipng.
    
    Возвращает True в случае успешной компиляции и создания файла, иначе False.
    Работает кроссплатформенно при наличии установленного TeX Live / MiKTeX в PATH.
    """
    if not is_latex_available():
        return False

    out_path = Path(output_filename).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir = Path(temp_dir) if temp_dir else out_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    stem = f"temp_render_{os.getpid()}"
    tex_path = base_dir / f"{stem}.tex"
    dvi_path = base_dir / f"{stem}.dvi"

    template = (
        r"\documentclass[preview,border=4pt]{standalone}"
        r"\usepackage{amsmath}"
        r"\usepackage{amsfonts}"
        r"\usepackage{amssymb}"
        r"\usepackage{mathtools}"
        r"\usepackage{amscd}"
        r"\usepackage{wasysym}"
        r"\usepackage{stmaryrd}"
        r"\begin{document}$" + formula_string + r"$\end{document}"
    )

    try:
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(template)

        cmd_latex = [
            "latex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            str(tex_path.name),
        ]
        res_latex = subprocess.run(
            cmd_latex,
            cwd=str(base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if res_latex.returncode != 0 or not dvi_path.exists():
            return False

        cmd_dvipng = [
            "dvipng",
            "-q",
            "-T",
            "tight",
            "-D",
            str(dpi),
            "-o",
            str(out_path),
            str(dvi_path.name),
        ]
        res_dvipng = subprocess.run(
            cmd_dvipng,
            cwd=str(base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return res_dvipng.returncode == 0 and out_path.exists()
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        for ext in ["tex", "log", "aux", "dvi"]:
            f_clean = base_dir / f"{stem}.{ext}"
            if f_clean.exists():
                try:
                    f_clean.unlink()
                except OSError:
                    pass
