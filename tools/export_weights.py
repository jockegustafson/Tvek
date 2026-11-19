"""Writes tvek/weights/certnet_v3.npz + MANIFEST.json + cue_model_v3.json from the svtx files (models/certainty/certnet.pt: three
seeds, audio + prosody, trained on 600 tags; cue_model.pkl: StandardScaler + LogisticRegression on 14 utterance cues). The npz holds
per seed only the parameters the released model uses ("<seed>/<param>"), the JSON the cue model as plain numbers, so no pickle and no
scikit-learn are needed at run time.
  python tools/export_weights.py certnet.pt cue_model.pkl tvek/weights"""
import hashlib, io, json, pickle, sys
from pathlib import Path
import numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); cue = pickle.load(open(sys.argv[2], "rb")); o = Path(sys.argv[3])
UNUSED = ("t.", "xatt.", "gate.")
arr, seeds = {}, []
for k, sd in enumerate(ck["models"]):
    s = f"CertNet_v3_s{k + 1}"; seeds.append(s)
    for n, v in sd.items():
        if not n.startswith(UNUSED): arr[f"{s}/{n}"] = v.float().numpy()
buf = io.BytesIO(); np.savez(buf, **arr); raw = buf.getvalue(); (o / "certnet_v3.npz").write_bytes(raw)
m = cue["model"]; sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
cj = {"features": cue["features"], "classes": [int(c) for c in lr.classes_], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
      "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}
(o / "cue_model_v3.json").write_text(json.dumps(cj))
(o / "MANIFEST.json").write_text(json.dumps({"model": "certnet_v3", "date": "2026-11-14", "seeds": seeds, "pcols": ck["pcols"],
    "npz_sha256": hashlib.sha256(raw).hexdigest(), "cue_sha256": hashlib.sha256((o / "cue_model_v3.json").read_bytes()).hexdigest(),
    "audio_model": "KBLab/wav2vec2-large-voxrex-swedish", "layers": [3, 6, 9, 12, 15, 18],
    "trained_on": "12,500 face-to-face-dialogue utterances: soft targets (text-LLM votes, a cue model, an audio-LLM teacher) + 600 tags by ear (weight 20)",
    "evaluation": "5-fold CV by session on the tags, certain vs uncertain (430): AUC 0.812, 0.830 with the cue model"}, indent=1))
print("wrote", o, round(len(raw) / 1e6, 1), "MB,", len(seeds), "seeds")
