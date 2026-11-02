"""Qwen3-ASR on Apple Silicon, through MLX (mlx-qwen3-asr) on the Metal GPU.

The CUDA-graph decoder in engine.py does not exist on a Mac. mlx-qwen3-asr runs the same
Qwen/Qwen3-ASR-1.7B weights, with greedy speculative decoding: a small 0.6B draft model
proposes tokens and the 1.7B model verifies them. The draft only changes speed, never the
text, so the output matches plain greedy decoding of the 1.7B model.

transcribe() has the same contract as Engine._transcribe: the earlier transcript can be
forced as the start of the answer so only the new audio has to be decoded.
"""
import logging
import subprocess
from typing import Optional

import numpy as np

from . import config

log = logging.getLogger("engine")

SAMPLE_RATE = 16000
NUM_DRAFT_TOKENS = 8  # fastest of 4 and 8 on a 3-minute sample on an M5 Pro

# The languages Qwen3-ASR supports, as listed in qwen_asr.inference.utils.SUPPORTED_LANGUAGES
# (qwen-asr 0.0.6). mlx-qwen3-asr has no complete list of its own.
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
    "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
    "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]


def _chip_name() -> str:
    try:
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=2).stdout.strip()
        return name or "Apple Silicon"
    except Exception:
        return "Apple Silicon"


class MLXBackend:
    def __init__(self):
        import mlx.core as mx
        from mlx_qwen3_asr import Session

        self.mx = mx
        self.dtype = mx.float16
        model = str(config.MODEL_PATH) if config.MODEL_PATH.exists() else config.MODEL_ID
        log.info("Loading %s (MLX)", model)
        self.session = Session(model, dtype=self.dtype)
        self.draft = None
        if config.DRAFT_MODEL:
            try:
                self.draft = Session(config.DRAFT_MODEL, dtype=self.dtype).model
            except Exception:
                log.exception("Draft model %s unavailable, decoding without it", config.DRAFT_MODEL)
        self.languages = list(SUPPORTED_LANGUAGES)
        self.chip = _chip_name()

    @property
    def decoder_name(self) -> str:
        return "MLX, speculative decoding" if self.draft is not None else "MLX"

    def _rollback(self, raw: str, k: int = 5) -> str:
        """Drop the last k tokens so the model can revise the end of what it said before."""
        tok = self.session.tokenizer
        ids = tok.encode(raw)
        for end in range(len(ids) - k, 0, -1):
            prefix = tok.decode(ids[:end])
            if "�" not in prefix:  # don't cut a multi-byte character in half
                return prefix
        return ""

    def transcribe(self, wav: np.ndarray, language: Optional[str], context: str,
                   continue_from: str = "", new_s: Optional[float] = None,
                   rollback: int = 5) -> tuple[str, str, str]:
        from mlx_qwen3_asr.audio import compute_features
        from mlx_qwen3_asr.generate import (GenerationConfig, generate_speculative_with_info,
                                            generate_with_info)
        from mlx_qwen3_asr.tokenizer import _detect_and_fix_repetitions, parse_asr_output

        mx, model, tok = self.mx, self.session.model, self.session.tokenizer
        language = language if language in self.languages else None
        wav = np.asarray(wav, dtype=np.float32)
        if wav.shape[0] < SAMPLE_RATE // 2:
            wav = np.pad(wav, (0, SAMPLE_RATE // 2 - wav.shape[0]))

        mel, feature_lens = compute_features(wav)
        audio_features, _ = model.audio_tower(mel.astype(self.dtype), feature_lens)
        prompt = tok.build_prompt_tokens(
            n_audio_tokens=audio_features.shape[1], language=language, context=context)
        # Continuing: the earlier transcript (minus its last `rollback` tokens) is forced as
        # the start of the answer, so only the new audio has to be decoded.
        prefix = ""
        if continue_from:
            prefix = self._rollback(continue_from, rollback) if rollback else continue_from
        if prefix:
            prompt = prompt + tok.encode(prefix)
        seconds = new_s if prefix and new_s is not None else wav.shape[0] / SAMPLE_RATE
        # Generous token budget, but bounded so a repetition loop can't run long.
        max_new = int(min(768, (48 if prefix else 32) + max(0.0, seconds) * 10))

        input_ids = mx.array([prompt])
        positions = mx.arange(input_ids.shape[1])[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)
        gen = GenerationConfig(max_new_tokens=max_new, temperature=0.0,
                               num_draft_tokens=NUM_DRAFT_TOKENS)
        if self.draft is None:
            result = generate_with_info(model=model, input_ids=input_ids,
                                        audio_features=audio_features,
                                        position_ids=position_ids, config=gen)
        else:
            draft_features, _ = self.draft.audio_tower(mel.astype(self.dtype), feature_lens)
            result = generate_speculative_with_info(
                model=model, draft_model=self.draft, input_ids=input_ids,
                audio_features=audio_features, draft_audio_features=draft_features,
                position_ids=position_ids, config=gen)
        raw = prefix + _detect_and_fix_repetitions(tok.decode(result.tokens))  # raw[len(prefix):] is new
        lang, text = parse_asr_output(raw, user_language=language)
        del mel, feature_lens, audio_features, input_ids, positions, position_ids, result
        mx.clear_cache()
        return lang, text.strip(), raw

    def info(self) -> dict:
        d = {"device": self.chip}
        try:
            d["vram_gb"] = round(self.mx.get_active_memory() / 1e9, 1)
        except Exception:
            pass
        return d
