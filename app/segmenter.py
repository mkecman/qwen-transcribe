"""Splits a 16 kHz audio stream into utterances at pauses, using Silero VAD.

Used for live dictation (fed ~100 ms at a time) and for uploaded files (probabilities
computed in one batched pass), so both modes cut speech the same way.
"""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

SAMPLE_RATE = 16000
FRAME = 512  # Silero's window at 16 kHz (32 ms)
FRAME_MS = FRAME * 1000 / SAMPLE_RATE


def load_vad():
    from silero_vad import load_silero_vad
    return load_silero_vad()


@dataclass
class Segment:
    audio: np.ndarray
    start_s: float  # position in the stream
    end_s: float


@dataclass
class SpeechStart:
    start_s: float


Event = Union[SpeechStart, Segment]


@torch.no_grad()
def speech_probs(wav: np.ndarray, vad, streams: int = 32) -> np.ndarray:
    """Per-frame speech probability for a whole file.

    The audio is cut into `streams` slices that run through Silero as one batch, which is
    several times faster than frame-by-frame. State resets only at the slice seams.
    """
    n = wav.shape[0] // FRAME
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    streams = max(1, min(streams, n))
    per = -(-n // streams)
    x = np.zeros(streams * per * FRAME, dtype=np.float32)
    x[: n * FRAME] = wav[: n * FRAME]
    x = x.reshape(streams, per, FRAME)
    vad.reset_states()
    probs = np.empty((streams, per), dtype=np.float32)
    for j in range(per):
        probs[:, j] = vad(torch.from_numpy(x[:, j, :]), SAMPLE_RATE).squeeze(-1).numpy()
    vad.reset_states()
    return probs.reshape(-1)[:n]


class Segmenter:
    def __init__(self, vad=None, pause_ms: int = 700, threshold: float = 0.5,
                 neg_threshold: float = 0.35, pre_roll_ms: int = 300, tail_ms: int = 250,
                 min_speech_ms: int = 200, soft_max_s: float = 20.0, hard_max_s: float = 30.0):
        self.vad = vad if vad is not None else load_vad()
        self.vad.reset_states()
        self.threshold, self.neg_threshold = threshold, neg_threshold
        self.pause_frames = max(1, round(pause_ms / FRAME_MS))
        # Past the soft limit a much shorter pause is enough to cut, so long monologues
        # still split at natural breaks instead of mid-word at the hard limit.
        self.short_pause_frames = max(1, round(200 / FRAME_MS))
        self.pre_roll = round(pre_roll_ms / FRAME_MS)
        self.tail = round(tail_ms / FRAME_MS)
        self.min_speech = round(min_speech_ms / FRAME_MS)
        self.soft_max = round(soft_max_s * 1000 / FRAME_MS)
        self.hard_max = round(hard_max_s * 1000 / FRAME_MS)

        self._pending = np.zeros(0, dtype=np.float32)
        self._frames_seen = 0
        self._history: list[np.ndarray] = []  # recent non-speech frames (pre-roll)
        self._seg: list[np.ndarray] = []
        self._seg_start = 0
        self._speech_frames = 0
        self._silence = 0
        self.in_speech = False

    @property
    def position_s(self) -> float:
        return self._frames_seen * FRAME / SAMPLE_RATE

    def current_audio(self) -> Optional[np.ndarray]:
        """Audio of the utterance in progress (for live previews)."""
        if not self.in_speech or not self._seg:
            return None
        return np.concatenate(self._seg)

    @torch.no_grad()
    def feed(self, samples: np.ndarray) -> list[Event]:
        """Live use: feed any amount of audio, get back speech starts and finished utterances."""
        out: list[Event] = []
        buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n = buf.shape[0] // FRAME
        self._pending = buf[n * FRAME:]
        for i in range(n):
            frame = buf[i * FRAME:(i + 1) * FRAME]
            prob = float(self.vad(torch.from_numpy(frame), SAMPLE_RATE).item())
            ev = self._process(frame, prob)
            if ev is not None:
                out.append(ev)
        return out

    def segment_file(self, wav: np.ndarray) -> list[Segment]:
        """Offline use: split a whole file into utterances."""
        probs = speech_probs(wav, self.vad)
        out = []
        for i, prob in enumerate(probs):
            ev = self._process(wav[i * FRAME:(i + 1) * FRAME], float(prob))
            if isinstance(ev, Segment):
                out.append(ev)
        last = self.flush()
        if last is not None:
            out.append(last)
        return out

    def _process(self, frame: np.ndarray, prob: float) -> Optional[Event]:
        self._frames_seen += 1
        if not self.in_speech:
            if prob >= self.threshold:
                self.in_speech = True
                self._seg = (self._history[-self.pre_roll:] if self.pre_roll else []) + [frame]
                self._seg_start = self._frames_seen - len(self._seg)
                self._speech_frames, self._silence = 1, 0
                self._history = []
                return SpeechStart(self._seg_start * FRAME / SAMPLE_RATE)
            self._history.append(frame)
            if len(self._history) > self.pre_roll:
                self._history.pop(0)
            return None

        self._seg.append(frame)
        if prob >= self.threshold:
            self._speech_frames += 1
            self._silence = 0
        elif prob < self.neg_threshold:
            self._silence += 1
        # between thresholds: neither speech nor silence (hysteresis)

        length = len(self._seg)
        needed = self.short_pause_frames if length >= self.soft_max else self.pause_frames
        if self._silence >= needed:
            return self._close(trim=self._silence)
        if length >= self.hard_max:
            return self._close(trim=0)
        return None

    def _close(self, trim: int) -> Optional[Segment]:
        keep = len(self._seg) - max(0, trim - self.tail)
        frames = self._seg[:keep]
        speech = self._speech_frames
        start = self._seg_start
        # carry the trimmed silence over as pre-roll for the next utterance
        self._history = self._seg[keep:][-self.pre_roll:] if self.pre_roll else []
        self._seg, self.in_speech = [], False
        self._speech_frames = self._silence = 0
        if speech < self.min_speech or not frames:
            return None
        audio = np.concatenate(frames)
        start_s = start * FRAME / SAMPLE_RATE
        return Segment(audio=audio, start_s=start_s, end_s=start_s + audio.shape[0] / SAMPLE_RATE)

    def flush(self) -> Optional[Segment]:
        """End of stream: close any utterance in progress."""
        if self._pending.shape[0] and self.in_speech:
            self._seg.append(self._pending)
        self._pending = np.zeros(0, dtype=np.float32)
        if not self.in_speech:
            return None
        return self._close(trim=self._silence)
