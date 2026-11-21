"""Tvek: how certain a Swedish speaker sounds, per utterance, with a running curve per word (CertNet v7; CertNet v3 with model="v3").
  det = CertaintyDetector()                                   # cuda, mps or cpu, whichever is there; model="v3" for the 1.0/1.1 model
  det.score_file("interview.wav", words, channel=0)           # words: [(word, start, end)] of ONE speaker
  det.score(x16, utterances)                                  # x16: 16 kHz mono; utterances: [[(start, end, word), ...], ...]
Utterances: the words split where the pause between two words is >= gap (0.8 s). Prosody is z-scored over all utterances of the call,
so give all of a speaker's words in one call (about 10 utterances or more)."""
import hashlib
import io
import json
import re
import warnings
from importlib import resources

import numpy as np
import torch

from .model import CL, NOZ, SR, Features, Net


def _load(model=None, weights_dir=None):
    """model None = the default (MANIFEST.json, CertNet v7), "v3" = MANIFEST_certnet_v3.json; the manifest names the weight files;
    weights_dir: another folder with a MANIFEST.json and the files it names (e.g. a model exported by tools/export_weights.py)"""
    if weights_dir is not None:
        from pathlib import Path
        ref = Path(weights_dir); model = None
    else: ref = resources.files("tvek") / "weights"
    man = json.loads((ref / ("MANIFEST.json" if model in (None, "v7", "certnet_v7") else f"MANIFEST_certnet_{model.replace('certnet_', '')}.json")).read_text())
    raw = (ref / man.get("npz", "certnet_v3.npz")).read_bytes(); cue = (ref / man.get("cue", "cue_model_v3.json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != man["npz_sha256"] or hashlib.sha256(cue).hexdigest() != man["cue_sha256"]:
        raise RuntimeError("tvek weights do not match MANIFEST.json")
    z = np.load(io.BytesIO(raw))
    seeds = [{k[len(s) + 1:]: torch.from_numpy(z[k].copy()) for k in z.files if k.startswith(s + "/")} for s in man["seeds"]]
    return man, seeds, json.loads(cue)


class CueModel:
    """the packaged cue model (StandardScaler + multinomial LogisticRegression) as plain numbers"""
    def __init__(self, d):
        self.f = d["features"]; self.mu, self.sc = np.array(d["mean"]), np.array(d["scale"])
        self.W, self.b = np.array(d["coef"]), np.array(d["intercept"])

    def proba(self, cues):
        z = self.W @ ((np.array([cues[f] for f in self.f]) - self.mu) / self.sc) + self.b
        e = np.exp(z - z.max()); return e / e.sum()


MAX_WORDS, MAX_S = 40, 20.0          # the training utterances: 3-40 words; longer stretches are split at their longest pause


def _cap(u):
    if len(u) <= MAX_WORDS and u[-1][1] - u[0][0] <= MAX_S or len(u) < 2: return [u]
    k = max(range(1, len(u)), key=lambda i: u[i][0] - u[i - 1][1] - 1e-3 * abs(i - len(u) / 2))
    return _cap(u[:k]) + _cap(u[k:])


def split_utterances(words, gap=0.8):
    """[(word, start, end)] -> [[(start, end, word)]]: words with letters or digits only, in time order, split at pauses >= gap s,
    and stretches over 40 words or 20 s split again at their longest pause (as the transcription engine segments)"""
    ws = sorted((float(s), float(e), str(w)) for w, s, e in words if w and re.search(r"\w", str(w)) and float(e) > float(s))
    out, cur = [], []
    for x in ws:
        if cur and x[0] - cur[-1][1] >= gap: out.append(cur); cur = []
        cur.append(x)
    if cur: out.append(cur)
    return [p for u in out for p in _cap(u)]


def load_audio(path, channel=None):
    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y[:, channel] if channel is not None and channel < y.shape[1] else y.mean(1)
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr); y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


class CertaintyDetector:
    def __init__(self, device=None, model=None, weights_dir=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.manifest, seeds, cue = _load(model, weights_dir); self.pcols = self.manifest["pcols"]
        self.nets = []
        for sd in seeds:
            n = Net(len(self.pcols)); n.load_state_dict(sd); self.nets.append(n.to(self.device).eval())
        self.cue = CueModel(cue); self.feat = Features(self.device)

    @torch.inference_mode()
    def score(self, x, utterances):
        """x: the speaker's channel, 16 kHz mono float32; utterances: [[(start, end, word)]] of ONE speaker in time order
        -> per utterance {start, end, text, label, p: {certain, uncertain, no_claim}, p_net, curve: [[3 probs] per word], cues}"""
        if len(utterances) < 10:
            warnings.warn(f"{len(utterances)} utterances: prosody is z-scored over the utterances given, so fewer than ~10 make it noisy")
        starts = [u[0][0] for u in utterances]; F = []
        for k, u in enumerate(utterances):
            nxt = next((s for s in starts[k + 1:] if s > u[-1][1] + 0.02), None); F.append(self.feat.features(x, u, nxt))
        allp = np.concatenate([p for _, p, _ in F]); zc = [k for k, c in enumerate(self.pcols) if c not in NOZ]
        mu, sd = allp[:, zc].mean(0), allp[:, zc].std(0) + 1e-3; out = []
        for (V, P, cues), u in zip(F, utterances):
            P = P.copy(); P[:, zc] = np.clip((P[:, zc] - mu) / sd, -6, 6)
            v = torch.from_numpy(V)[None].to(self.device); p = torch.from_numpy(P)[None].to(self.device)
            c = np.mean([torch.softmax(n(v, p)[0].float(), -1).cpu().numpy() for n in self.nets], 0)
            q = self.cue.proba(cues); pu = (c[-1] + q) / 2
            out.append({"start": u[0][0], "end": u[-1][1], "text": " ".join(w for _, _, w in u), "label": CL[int(pu.argmax())],
                        "p": {k: round(float(v), 3) for k, v in zip(CL, pu)}, "p_net": [round(float(v), 3) for v in c[-1]],
                        "curve": [[round(float(v), 3) for v in row] for row in c[:-1]], "cues": {k: round(v, 3) for k, v in cues.items()}})
        return out

    def score_file(self, path, words, channel=None, gap=0.8):
        """path: any audio soundfile reads; words: [(word, start, end)] of the speaker on `channel` (None = mix the channels)"""
        return self.score(load_audio(path, channel), split_utterances(words, gap))
