// Captures microphone audio as 16 kHz mono Int16 PCM, posted in 100 ms blocks.
// The AudioContext normally runs at 16 kHz already (Chromium resamples for us);
// otherwise this resamples with a light low-pass + linear interpolation.
const OUT_RATE = 16000;
const BLOCK = 1600;

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / OUT_RATE;
    this.alpha = this.step > 1 ? 1 / this.step : 1;
    this.lp = 0;
    this.t = 0;      // read position of the next output sample, relative to the current block
    this.last = 0;   // last filtered sample of the previous block
    this.tmp = new Float32Array(128);
    this.out = new Int16Array(BLOCK);
    this.n = 0;
    this.sq = 0;
  }

  emit(v) {
    v = Math.max(-1, Math.min(1, v));
    this.out[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
    this.sq += v * v;
    if (this.n === BLOCK) {
      const rms = Math.sqrt(this.sq / BLOCK);
      const level = Math.max(0, Math.min(1, (20 * Math.log10(rms + 1e-9) + 60) / 60));
      this.port.postMessage({ pcm: this.out.buffer, level }, [this.out.buffer]);
      this.out = new Int16Array(BLOCK);
      this.n = 0;
      this.sq = 0;
    }
  }

  process(inputs) {
    const x = inputs[0] && inputs[0][0];
    if (!x) return true;
    if (this.step === 1) {
      for (let i = 0; i < x.length; i++) this.emit(x[i]);
      return true;
    }
    const N = x.length;
    if (this.tmp.length < N) this.tmp = new Float32Array(N);
    const y = this.tmp;
    for (let i = 0; i < N; i++) {
      this.lp += this.alpha * (x[i] - this.lp);
      y[i] = this.lp;
    }
    while (this.t < N - 1) {
      const i = Math.floor(this.t);
      const f = this.t - i;
      const a = i < 0 ? this.last : y[i];
      this.emit(a + (y[i + 1] - a) * f);
      this.t += this.step;
    }
    this.t -= N;
    this.last = y[N - 1];
    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);
