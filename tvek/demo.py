"""tvek-demo: a local web page for Tvek (record or load speech -> each utterance coloured certain / uncertain / no claim, with the
running uncertainty curve over the words, on a spectrogram).

    tvek-demo [--port 8767] [--device cuda|mps|cpu] [--host 127.0.0.1] [--no-browser]

then open http://localhost:8767 (the microphone works on localhost). Nothing leaves the machine. Tvek needs word times: give a TextGrid
or a TSV of ONE speaker's words, or let the page get them from KB-Whisper (KBLab/kb-whisper-large, pip install "tvek[asr]", ~3 GB on first
use; its word times are rough, from cross-attention). Recordings over MAX_S = 300 s are cut.
  POST /analyze {pcm, sr, words?, tier?} -> spectrogram, words, utterances (label, p, curve per word)"""
import argparse, base64, json, os, tempfile, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import numpy as np

from .detector import split_utterances
from .model import SR

MAX_S = 300.0
DET = None; ASR = None; LOCK = threading.Lock()


def to16k(y, sr):
    y = np.asarray(y, np.float32)
    if int(sr) != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, int(sr)); y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return y[:int(MAX_S * SR)]


def spectrogram(y, n_fft=512, hop=160):
    if len(y) < n_fft: y = np.pad(y, (0, n_fft - len(y)))
    w = np.hanning(n_fft).astype(np.float32); n = 1 + (len(y) - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(y, (n, n_fft), (y.strides[0] * hop, y.strides[0])) * w
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-6); top = S.max(); S = np.clip((S - (top - 80)) / 80, 0, 1)
    return {"frames": int(n), "bins": int(S.shape[1]), "data": base64.b64encode((S * 255).astype(np.uint8).tobytes()).decode()}


def asr_words(y):
    """word times from KB-Whisper (transformers pipeline, return_timestamps="word"), 30 s chunks"""
    global ASR
    if ASR is None:
        import torch
        from transformers import pipeline
        dev = DET.device; dt = torch.float16 if dev in ("cuda", "mps") else torch.float32
        ASR = pipeline("automatic-speech-recognition", model="KBLab/kb-whisper-large", device=dev, torch_dtype=dt)
    r = ASR({"raw": y, "sampling_rate": SR}, return_timestamps="word", chunk_length_s=30, generate_kwargs={"language": "sv", "task": "transcribe"})
    out = []
    for c in r["chunks"]:
        a, b = c["timestamp"]
        if a is None: continue
        b = b if b is not None else a + 0.3; out.append((c["text"].strip(), float(a), float(max(b, a + 0.05))))
    return out


def analyze(req):
    y = to16k(np.frombuffer(base64.b64decode(req["pcm"]), np.float32), req.get("sr", SR))
    if req.get("words"):
        from .cli import read_words
        suffix = ".TextGrid" if "IntervalTier" in req["words"] else ".tsv"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f: f.write(req["words"])
        try: words, src = read_words(f.name, req.get("tier") or None), "word file"
        finally: os.unlink(f.name)
    else:
        with LOCK: words, src = asr_words(y), "KB-Whisper word times (rough)"
    dur = len(y) / SR; words = [w for w in words if float(w[2]) <= dur and float(w[1]) >= 0]    # words outside the audio are dropped
    U = split_utterances(words)
    with LOCK: res = DET.score(y, U) if U else []
    W = [{"w": w, "s": s, "e": e} for u in U for s, e, w in u]
    return {"dur": len(y) / SR, "spec": spectrogram(y), "source": src, "words": W,
            "utterances": [{k: r[k] for k in ("start", "end", "text", "label", "p", "curve")} | {"words": [[s, e, w] for s, e, w in u]} for r, u in zip(res, U)]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode(); self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"): self._send(200, (resources.files("tvek") / "demo.html").read_bytes(), "text/html; charset=utf-8")
        else: self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze": return self._send(404, "not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, json.dumps(analyze(req), ensure_ascii=False, default=float), "application/json")
        except SystemExit as ex: self._send(400, json.dumps({"error": str(ex)}), "application/json")
        except Exception as ex: self._send(500, json.dumps({"error": f"{type(ex).__name__}: {ex}"}), "application/json")


def main():
    global DET
    ap = argparse.ArgumentParser(description="Tvek demo page (local)")
    ap.add_argument("--port", type=int, default=8767); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--device")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    from . import CertaintyDetector
    print("loading Tvek (VoxRex, ~1.3 GB on first use) ...", flush=True)
    DET = CertaintyDetector(a.device)
    url = f"http://localhost:{a.port}"; print(f"Tvek demo ({DET.device}) at {url}  - Ctrl-C to stop", flush=True)
    if not a.no_browser: threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
