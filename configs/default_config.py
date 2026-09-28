from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class DataConfig:
    """Параметры предобработки и загрузки данных."""
    base_path: str = "./dataset/dataset_clean"
    img_height: int = 128
    img_width: int = 1024
    img_channels: int = 1
    norm_mean: Tuple[float, ...] = (0.0672,)
    norm_std: Tuple[float, ...] = (0.1625,)
    max_len: int = 250
    special_tokens: Tuple[str, ...] = ("<P>", "<S>", "<E>", "<UNK>")
    pad_token: str = "<P>"
    sos_token: str = "<S>"
    eos_token: str = "<E>"
    unk_token: str = "<UNK>"


@dataclass
class ModelConfig:
    """Параметры гибридной архитектуры CNN + Transformer Decoder."""
    vocab_size: int = 0  # Вычисляется динамически по custom_t2i.json
    d_model: int = 256
    nhead: int = 8
    num_decoder_layers: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 250
    version: int = 10

    # CNN Backbone параметры
    c_in: int = 1
    stem_channels: Tuple[int, int] = (32, 64)
    stage_channels: Tuple[int, int, int, int] = (128, 256, 256, 256)
    psa_channels: int = 256
    psa_heads: int = 4


@dataclass
class TrainingConfig:
    """Параметры цикла обучения модели."""
    num_epochs: int = 150
    batch_size: int = 32  # При None рассчитывается динамически по VRAM
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    weight_ratio: float = 5.0
    patience: int = 15
    lr_factor: float = 0.5
    lr_patience: int = 3
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    model_name: str = "im2latex_model"
    checkpoints_dir: str = "./checkpoints"


@dataclass
class EvaluationConfig:
    """Параметры тестирования и инференса."""
    beam_width: int = 3
    max_len: int = 250
    length_penalty_alpha: float = 0.7
    num_visual_samples: int = 10
    render_output_dir: str = "./test_renders"


@dataclass
class SystemConfig:
    """Системные параметры платформы и памяти."""
    cuda_alloc_conf: str = "expandable_segments:True"
    min_vram_gb: float = 0.5
    vram_margin_gb: float = 1.0


@dataclass
class GlobalConfig:
    """Глобальный контейнер конфигураций."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
