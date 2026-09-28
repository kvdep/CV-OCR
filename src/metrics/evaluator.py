from typing import Sequence, Any, Dict, Optional


def calculate_levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Вычисляет расстояние Левенштейна между двумя последовательностями (символов или токенов).
    
    Реализует алгоритм Вагнера-Фишера с оптимизацией по памяти O(min(N, M)).
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    if n > m:
        seq1, seq2 = seq2, seq1
        n, m = m, n

    current_row = list(range(n + 1))
    for j in range(1, m + 1):
        previous_row = current_row
        current_row = [j] + [0] * n
        for i in range(1, n + 1):
            add = previous_row[i] + 1
            delete = current_row[i - 1] + 1
            change = (
                previous_row[i - 1]
                if seq1[i - 1] == seq2[j - 1]
                else previous_row[i - 1] + 1
            )
            current_row[i] = min(add, delete, change)
    return current_row[n]


def calculate_ecdm(pred_seq: Sequence[Any], target_seq: Sequence[Any]) -> float:
    """Вычисляет метрику сходства ECDM (Edit-based Distance Metric) для одного образца.
    
    Формула:
        ECDM = max(0.0, 1.0 - Levenshtein(pred, target) / max(|pred|, |target|, 1))
    """
    lev = calculate_levenshtein_distance(pred_seq, target_seq)
    max_len = max(len(pred_seq), len(target_seq), 1)
    return max(0.0, 1.0 - (lev / max_len))


class EvaluationMetrics:
    """Накопитель и агрегатор метрик качества OCR распознавания."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_samples: int = 0
        self.total_exact_matches: int = 0
        self.sum_char_ecdm: float = 0.0
        self.sum_token_ecdm: float = 0.0
        self.total_char_levenshtein: int = 0
        self.total_target_char_length: int = 0
        self.total_token_levenshtein: int = 0
        self.total_target_token_length: int = 0
        self.total_loss: float = 0.0

    def update(
        self,
        pred_str: str,
        target_str: str,
        loss_val: Optional[float] = None,
    ) -> Dict[str, float]:
        """Обновляет статистику парой (предсказание, эталон)."""
        pred_str = pred_str.strip()
        target_str = target_str.strip()

        pred_tokens = pred_str.split()
        target_tokens = target_str.split()

        char_lev = calculate_levenshtein_distance(pred_str, target_str)
        char_max = max(len(pred_str), len(target_str), 1)
        char_ecdm = max(0.0, 1.0 - (char_lev / char_max))

        token_lev = calculate_levenshtein_distance(pred_tokens, target_tokens)
        token_max = max(len(pred_tokens), len(target_tokens), 1)
        token_ecdm = max(0.0, 1.0 - (token_lev / token_max))

        is_exact = int(pred_str == target_str)

        self.total_samples += 1
        self.total_exact_matches += is_exact
        self.sum_char_ecdm += char_ecdm
        self.sum_token_ecdm += token_ecdm
        self.total_char_levenshtein += char_lev
        self.total_target_char_length += len(target_str)
        self.total_token_levenshtein += token_lev
        self.total_target_token_length += len(target_tokens)

        if loss_val is not None:
            self.total_loss += loss_val

        return {
            "char_ecdm": char_ecdm,
            "token_ecdm": token_ecdm,
            "char_lev": char_lev,
            "token_lev": token_lev,
            "is_exact": is_exact,
        }

    def compute_summary(self) -> Dict[str, float]:
        """Вычисляет итоговые макро- и микро-агрегированные метрики."""
        n = max(self.total_samples, 1)
        return {
            "total_samples": float(self.total_samples),
            "exact_match_rate": (self.total_exact_matches / n) * 100.0,
            "macro_char_ecdm": (self.sum_char_ecdm / n) * 100.0,
            "macro_token_ecdm": (self.sum_token_ecdm / n) * 100.0,
            "micro_cer": (
                self.total_char_levenshtein / max(1, self.total_target_char_length)
            ) * 100.0,
            "micro_ter": (
                self.total_token_levenshtein / max(1, self.total_target_token_length)
            ) * 100.0,
            "mean_loss": (self.total_loss / n) if self.total_loss > 0 else 0.0,
        }
