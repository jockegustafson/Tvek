"""Tvek model code: CertNet v3 of the KTH svtx toolkit (svtx/uncertainty.py), unchanged in what it computes.
Constants, the network, the lexicon flags, and the per-utterance features (VoxRex states, eGeMAPS prosody, utterance cues)."""
import numpy as np
import torch
import torch.nn as nn

CL = ["certain", "uncertain", "no_claim"]; LAYERS = [3, 6, 9, 12, 15, 18]; TAIL = 1.2; SR = 16000
FN = set("men och att som så eller för i på med till av om när då den det en ett de du jag han hon vi man ni är var har ska kan vill "
         "kunde skulle lite bara liksom typ alltså ju väl nog ändå också inte".split())
FILL = {"eh", "öh", "ehm", "öhm", "äh", "ähm", "mm", "hm", "eeh", "öhh", "eh-", "öh-"}
HEDGE = set("kanske nog väl antagligen troligen troligtvis möjligen eventuellt typ liksom ungefär cirka verkar tror antar känns "
            "osäker borde nånting någonting nåt sådär".split())
BOOST = set("ju faktiskt absolut verkligen definitivt självklart säkert precis exakt alltid aldrig klart".split())
MULTI = [("jag", "tror"), ("vet", "inte"), ("eller", "hur"), ("eller", "nåt"), ("eller", "så"), ("jag", "antar"), ("tror", "jag")]
NOZ = {"is_tail", "fn", "fill", "hedge", "boost", "pos_from_end"}


class Net(nn.Module):
    """CertNet v3: per token (each word + the TAIL token) a learned softmax mix of 6 VoxRex layers (word mean and word end) and the
    prosody vector, each projected to 256 and added, a 2-layer causal transformer, 3 classes per token. (The research version also
    had a text stream; it lowered the score and is not part of the released model.)"""
    def __init__(self, P, d=256, drop=0.2, use=("audio", "pros")):
        super().__init__(); self.use = use
        self.lw = nn.Parameter(torch.zeros(2, 6)); self.a = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d), nn.GELU(), nn.Dropout(drop))
        self.p = nn.Sequential(nn.Linear(P, 128), nn.GELU(), nn.Linear(128, d), nn.Dropout(drop))
        self.pos = nn.Embedding(64, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, drop, batch_first=True, norm_first=True), 2)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 3))

    def forward(self, vox, pros):
        T = vox.shape[1]; causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=vox.device), 1)
        w = torch.softmax(self.lw, -1); a = self.a(torch.einsum("btkld,kl->btkd", vox.float(), w).reshape(1, T, -1))
        h = a + self.p(pros) + self.pos(torch.arange(T, device=vox.device).clamp(max=63))[None]
        return self.head(self.enc(h, mask=causal))


def lex(words):
    w = [x.lower().strip(".,?!") for x in words]; hedge = [float(x in HEDGE) for x in w]
    for i in range(len(w) - 1):
        if (w[i], w[i + 1]) in MULTI: hedge[i] = hedge[i + 1] = 1.0
    return [float(x in FN) for x in w], [float(x in FILL) for x in w], hedge, [float(x in BOOST) for x in w]




class Features:
    """VoxRex + openSMILE feature extraction for one speaker's channel (16 kHz mono)"""

    def __init__(self, device):
        import opensmile
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.dev = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
        self.vox = Wav2Vec2Model.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish").to(device).eval()
        if device not in ("cpu", "mps"): self.vox = self.vox.half()
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

    @torch.inference_mode()
    def features(self, x, utt, nxt):
        """one utterance [(s, e, w)], the speaker's next word start (or None) -> vox (n+1, 2, 6, 1024), pros (n+1, P), cues"""
        words = [w for _, _, w in utt]; times = [(s, e) for s, e, _ in utt]; t0, t1 = times[0][0], times[-1][1]
        tail_end = min(t1 + TAIL, nxt) if nxt else t1 + TAIL; a0 = max(0.0, t0 - 0.3)
        x16 = x[int(a0 * SR):int((t1 + TAIL) * SR)].astype(np.float32)
        spans = [(a - a0, b - a0) for a, b in times] + [(t1 - a0, max(t1 + 0.04, tail_end) - a0)]
        iv = self.fe(x16, sampling_rate=SR, return_tensors="pt").input_values.to(self.dev)
        hs = self.vox(iv.half() if self.dev not in ("cpu", "mps") else iv, output_hidden_states=True).hidden_states
        H = torch.stack([hs[k][0] for k in LAYERS]).float().cpu().numpy(); tf = np.arange(H.shape[1]) * 0.02 + 0.01
        V = np.zeros((len(spans), 2, 6, 1024), np.float32)
        for j, (a, b) in enumerate(spans):
            m = (tf >= a) & (tf < b); m = m if m.any() else np.abs(tf - (a + b) / 2) == np.abs(tf - (a + b) / 2).min()
            e = (tf >= max(a, b - 0.1)) & (tf < b); e = e if e.any() else m
            V[j, 0] = H[:, m].mean(1); V[j, 1] = H[:, e].mean(1)
        df = self.smile.process_signal(x16, SR); lld = list(df.columns); L = df.values.astype(np.float32)
        lt = np.array([s.total_seconds() for s in df.index.get_level_values("start")]) + 0.01
        f0 = L[:, lld.index("F0semitoneFrom27.5Hz_sma3nz")]; ld = L[:, lld.index("Loudness_sma3")]; vo = f0 > 0
        fn, fill, hedge, boost = lex(words); n = len(words)
        per = [(b - a) / max(1, len(w)) for (a, b), w in zip(times, words)]; mper = float(np.mean(per)) if per else 1.0
        P = []
        for j, (a, b) in enumerate(spans):
            m = (lt >= a) & (lt < b)
            if not m.any(): m = np.abs(lt - (a + b) / 2) == np.abs(lt - (a + b) / 2).min()
            mv = m & vo; f0v = f0[mv]; slope = float(np.polyfit(lt[mv], f0v, 1)[0]) if mv.sum() >= 5 else 0.0; tail = j == n
            gap = (spans[j + 1][0] - b) if j < n - 1 else (tail_end - t1 if j == n - 1 else 0.0)
            P.append(list(L[m].mean(0)) + [float(f0v.std()) if len(f0v) > 2 else 0.0, float(ld[m].std()), float(mv.sum() / max(1, m.sum())), b - a,
                     (np.log(per[j] / mper) if not tail and mper > 0 and per[j] > 0 else 0.0), slope, gap, (n - j) / max(1, n), float(tail),
                     0.0 if tail else fn[j], 0.0 if tail else fill[j], 0.0 if tail else hedge[j], 0.0 if tail else boost[j]])
        # utterance cues (the cue model)
        tl = (lt >= t1 - a0) & (lt < t1 + TAIL - a0); cov = np.zeros(len(lt), bool)
        if nxt and nxt <= t1 + TAIL: cov |= lt >= nxt - a0
        lv = np.array([20 * np.log10(np.sqrt(np.mean(x16[max(0, int(q * SR) - 80):int(q * SR) + 80] ** 2)) + 1e-9) for q in lt])
        wa, wb = times[-1][0] - a0, t1 - a0; vi = [k for k in np.where((lt >= wa) & (lt <= wb))[0] if vo[k]]; run_v, tail_db = 0, -30.0
        if vi:
            i = j = vi[-1]; gap = 0; bidx = int(np.searchsorted(lt, wb))
            while j + 1 < len(lt) and gap <= 4:
                j += 1; gap = 0 if vo[j] else gap + 1
                if vo[j]: i = j
            run_v = max(0, i - bidx)
            if run_v > 2: tail_db = float(np.median(lv[bidx:i + 1]) - (np.median(lv[vo]) if vo.any() else -40.0))
        uv = (lt < t1 - a0) & vo; tv, sv = lt[uv], f0[uv]; m4 = tv >= (tv[-1] - 0.4) if len(tv) else np.zeros(0, bool)
        last = words[-1].lower().strip(".,?!"); ll = float(np.log(per[-1] / mper)) if n >= 2 and mper > 0 else 0.0
        cues = {"last_fn": float(last in FN), "last_fill": float(last in FILL), "trail_voice": float((vo & tl & ~cov).sum() * 0.01),
                "trail_sil": float(min(4.0, (nxt - t1) if nxt else 4.0)), "dur": t1 - t0, "rate": sum(map(len, words)) / max(1e-3, t1 - t0),
                "end_slope": float(np.polyfit(tv[m4], sv[m4], 1)[0]) if m4.sum() >= 5 else 0.0, "f0_sd": float(sv.std()) if len(sv) > 2 else 4.0,
                "last_len": ll, "tail_s": run_v * 0.01, "tail_db": tail_db, "fn_len": float(last in FN) * ll,
                "n_hedge": float(sum(hedge)), "n_boost": float(sum(boost))}
        return V, np.nan_to_num(np.array(P, np.float32)), cues

