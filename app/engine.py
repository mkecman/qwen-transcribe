"""Qwen3-ASR inference: model loading, a CUDA-graph greedy decoder, and one GPU worker thread.

All GPU work runs on a single thread, pulled from a priority queue, so live dictation
(finals, then partials) always jumps ahead of file transcription.
"""
import asyncio
import itertools
import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import config

log = logging.getLogger("engine")

SAMPLE_RATE = 16000
PRIORITY_FINAL, PRIORITY_PARTIAL, PRIORITY_FILE = 0, 1, 2


class PromptTooLong(Exception):
    pass


class GraphDecoder:
    """Greedy single-stream decoder for the Qwen3-ASR text decoder.

    Prefill runs through the regular transformers code. Each decode step is replayed from a
    CUDA graph, which removes the per-kernel launch overhead that dominates on Windows.
    One graph is captured per KV-length bucket so attention only scans what it needs.
    """

    BUCKETS = (256, 512, 1024, 2048, 4096)

    def __init__(self, asr):
        from transformers import StaticCache
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import apply_rotary_pos_emb

        self._rope = apply_rotary_pos_emb
        self.asr = asr
        self.thinker = asr.model.thinker
        self.proc = asr.processor
        text_model = self.thinker.model
        self.embed = text_model.embed_tokens
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary = text_model.rotary_emb
        self.lm_head = self.thinker.lm_head
        self.dev = self.thinker.device
        self.dtype = self.thinker.dtype
        self.eos = set(asr.model.generation_config.eos_token_id)

        tc = self.thinker.config.get_text_config()
        self.head_dim = tc.head_dim
        self.max_len = self.BUCKETS[-1]
        self.cache = StaticCache(config=self.thinker.config, max_cache_len=self.max_len)
        self.cache.early_initialization(1, tc.num_key_value_heads, tc.head_dim, self.dtype, self.dev)
        self.kv = [(layer.keys, layer.values) for layer in self.cache.layers]

        self.tok = torch.zeros((1, 1), dtype=torch.long, device=self.dev)
        self.pos = torch.zeros((1,), dtype=torch.long, device=self.dev)
        self.arange = torch.arange(self.max_len, device=self.dev)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    @torch.no_grad()
    def _step(self, bucket: int):
        hd = self.head_dim
        h = self.embed(self.tok)
        p = self.pos.view(1, 1, 1).expand(3, 1, 1)
        cos, sin = self.rotary(h, p)
        mask = (self.arange[:bucket] <= self.pos).view(1, 1, 1, bucket)
        for layer, (keys, values) in zip(self.layers, self.kv):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).view(1, 1, -1, hd)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(1, 1, -1, hd)).transpose(1, 2)
            v = attn.v_proj(x).view(1, 1, -1, hd).transpose(1, 2)
            q, k = self._rope(q, k, cos, sin)
            keys.index_copy_(2, self.pos, k)
            values.index_copy_(2, self.pos, v)
            o = F.scaled_dot_product_attention(
                q, keys[:, :, :bucket], values[:, :, :bucket],
                attn_mask=mask, scale=attn.scaling, enable_gqa=True,
            )
            h = h + attn.o_proj(o.transpose(1, 2).reshape(1, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.norm(h)
        nxt = torch.argmax(self.lm_head(h[:, -1, :]), dim=-1)
        self.tok.copy_(nxt.view(1, 1))
        self.pos.add_(1)

    def capture(self):
        pool = torch.cuda.graph_pool_handle()
        for bucket in self.BUCKETS:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.pos.zero_()
                    self._step(bucket)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self._step(bucket)
            self.graphs[bucket] = graph
        torch.cuda.synchronize()

    @torch.no_grad()
    def generate(self, wav: np.ndarray, prompt: str, max_new: int) -> str:
        inp = self.proc(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inp = inp.to(self.dev).to(self.dtype)
        prompt_len = inp["input_ids"].shape[1]
        max_new = min(max_new, self.max_len - prompt_len - 1)
        if max_new < 16:
            raise PromptTooLong(prompt_len)

        out = self.thinker(
            input_ids=inp["input_ids"],
            input_features=inp["input_features"],
            feature_attention_mask=inp["feature_attention_mask"],
            past_key_values=self.cache,
            cache_position=torch.arange(prompt_len, device=self.dev),
            use_cache=True,
        )
        token = int(torch.argmax(out.logits[0, -1]).item())
        ids = []
        if token not in self.eos:
            ids.append(token)
            self.tok.fill_(token)
            self.pos.fill_(prompt_len)
            for i in range(max_new - 1):
                position = prompt_len + i
                bucket = next(b for b in self.BUCKETS if position < b)
                self.graphs[bucket].replay()
                token = int(self.tok.item())
                if token in self.eos:
                    break
                ids.append(token)
        return self.proc.tokenizer.decode(ids, skip_special_tokens=True)


class Engine:
    def __init__(self):
        self.status = "loading"  # loading | ready | error
        self.error: Optional[str] = None
        self.fast_decoder = False
        self.languages: list[str] = []
        self.asr = None
        self.decoder: Optional[GraphDecoder] = None
        self._queue: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)

    def start(self):
        self._thread.start()

    # ---- worker thread -------------------------------------------------
    def _run(self):
        try:
            self._load()
            self.status = "ready"
        except Exception as e:  # keep the web UI alive so it can show the error
            log.exception("Model failed to load")
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            return
        while True:
            _, _, fn, args, fut = self._queue.get()
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args))
            except Exception as e:
                log.exception("GPU task failed")
                fut.set_exception(e)

    def _load(self):
        t0 = time.time()
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import SUPPORTED_LANGUAGES

        self.languages = list(SUPPORTED_LANGUAGES)
        log.info("Loading %s", config.MODEL_PATH)
        self.asr = Qwen3ASRModel.from_pretrained(
            str(config.MODEL_PATH), dtype=torch.bfloat16, device_map="cuda:0",
            max_inference_batch_size=8, max_new_tokens=512,
        )
        try:
            decoder = GraphDecoder(self.asr)
            decoder.capture()
            self.decoder, self.fast_decoder = decoder, True
        except Exception:
            log.exception("CUDA graph decoder unavailable, falling back to standard generation")
        # warm-up so the first real request is fast
        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), "English", "")
        log.info("Model ready in %.1fs (fast decoder: %s)", time.time() - t0, self.fast_decoder)

    # ---- inference -----------------------------------------------------
    def _prompt(self, context: str, language: Optional[str]) -> str:
        return self.asr._build_text_prompt(context=context, force_language=language)

    def _rollback(self, raw: str, k: int = 5) -> str:
        """Drop the last k tokens so the model can revise the end of what it said before."""
        tok = self.asr.processor.tokenizer
        ids = tok.encode(raw)
        for end in range(len(ids) - k, 0, -1):
            prefix = tok.decode(ids[:end])
            if "�" not in prefix:  # don't cut a multi-byte character in half
                return prefix
        return ""

    def _transcribe(self, wav: np.ndarray, language: Optional[str], context: str,
                    continue_from: str = "", new_s: Optional[float] = None,
                    rollback: int = 5) -> tuple[str, str, str]:
        from qwen_asr.inference.utils import detect_and_fix_repetitions, parse_asr_output

        language = language if language in self.languages else None
        wav = np.asarray(wav, dtype=np.float32)
        if wav.shape[0] < SAMPLE_RATE // 2:
            wav = np.pad(wav, (0, SAMPLE_RATE // 2 - wav.shape[0]))
        if self.decoder is not None:
            # Continuing: the earlier transcript (minus its last `rollback` tokens) is forced as
            # the start of the answer, so only the new audio has to be decoded.
            prefix = ""
            if continue_from:
                prefix = self._rollback(continue_from, rollback) if rollback else continue_from
            seconds = new_s if prefix and new_s is not None else wav.shape[0] / SAMPLE_RATE
            # Generous token budget, but bounded so a repetition loop can't run long.
            max_new = int(min(768, (48 if prefix else 32) + max(0.0, seconds) * 10))
            try:
                generated = self.decoder.generate(wav, self._prompt(context, language) + prefix, max_new)
                raw = prefix + detect_and_fix_repetitions(generated)  # raw[len(prefix):] is new
                lang, text = parse_asr_output(raw, user_language=language)
                return lang, text.strip(), raw
            except PromptTooLong:
                pass
        r = self.asr.transcribe(audio=(wav, SAMPLE_RATE), context=context, language=language)[0]
        raw = r.text if language else f"language {r.language}<asr_text>{r.text}"
        return r.language, r.text.strip(), raw

    def submit(self, priority: int, fn, *args) -> Future:
        fut: Future = Future()
        self._queue.put((priority, next(self._seq), fn, args, fut))
        return fut

    async def transcribe(self, wav: np.ndarray, language: Optional[str] = "English", context: str = "",
                         priority: int = PRIORITY_FINAL, continue_from: str = "",
                         new_s: Optional[float] = None, rollback: int = 5) -> tuple[str, str, str]:
        """Returns (detected_language, text, raw). Pass a previous `raw` as `continue_from`
        when `wav` extends the audio it was transcribed from; with `rollback=0` it is forced
        verbatim and `raw[len(continue_from):]` is the newly decoded text."""
        fut = self.submit(priority, self._transcribe, wav, language, context, continue_from, new_s, rollback)
        return await asyncio.wrap_future(fut)

    def info(self) -> dict:
        d = {"status": self.status, "error": self.error, "fast_decoder": self.fast_decoder,
             "languages": self.languages, "model": config.MODEL_PATH.name}
        if torch.cuda.is_available():
            d["device"] = torch.cuda.get_device_name(0)
            d["vram_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 1)
        return d


engine = Engine()
