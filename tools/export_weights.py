"""Writes tvek/weights/<name>.npz + cue_model_<name>.json + a manifest from svtx-style CertNet files (a .pt with the seed models, "pcols"
and "use"; a cue_model.pkl: StandardScaler + LogisticRegression on 14 utterance cues). The npz holds per seed only the parameters the
released model uses ("<seed>/<param>"), the JSON the cue model as plain numbers, so no pickle and no scikit-learn are needed at run time.
Models from the later trainer (cert_net5.py and after: CertNet v6, v7) name their single audio stream ("a.vox.*", "lw.vox"); those keys
are renamed to the released network's ("a.*", "lw"), which is otherwise identical.
  python tools/export_weights.py certnet.pt cue_model.pkl tvek/weights NAME MANIFEST_FILE INFO.json
INFO.json: {"date", "trained_on", "evaluation"} for the manifest. The default model's manifest is MANIFEST.json; others MANIFEST_<name>.json."""
import hashlib, io, json, pickle, re, sys
from pathlib import Path
import numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); cue = pickle.load(open(sys.argv[2], "rb")); o = Path(sys.argv[3])
name, man_file, info = sys.argv[4], sys.argv[5], json.load(open(sys.argv[6]))
UNUSED = ("t.", "xatt.", "gate.")
def rename(k): return re.sub(r"^a\.vox\.", "a.", re.sub(r"^lw\.vox$", "lw", k))
arr, seeds = {}, []
for k, sd in enumerate(ck["models"]):
    s = f"CertNet_{name.split('_')[-1]}_s{k + 1}"; seeds.append(s)
    for n, v in sd.items():
        if not n.startswith(UNUSED): arr[f"{s}/{rename(n)}"] = v.float().numpy()
buf = io.BytesIO(); np.savez(buf, **arr); raw = buf.getvalue(); (o / f"{name}.npz").write_bytes(raw)
m = cue["model"]; sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
cj = {"features": cue["features"], "classes": [int(c) for c in lr.classes_], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
      "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}
(o / f"cue_model_{name}.json").write_text(json.dumps(cj))
(o / man_file).write_text(json.dumps({"model": name, "npz": f"{name}.npz", "cue": f"cue_model_{name}.json", "date": info["date"], "seeds": seeds,
    "pcols": ck["pcols"], "npz_sha256": hashlib.sha256(raw).hexdigest(), "cue_sha256": hashlib.sha256((o / f"cue_model_{name}.json").read_bytes()).hexdigest(),
    "audio_model": "KBLab/wav2vec2-large-voxrex-swedish", "layers": [3, 6, 9, 12, 15, 18], "trained_on": info["trained_on"], "evaluation": info["evaluation"]}, indent=1))
print("wrote", o, name, round(len(raw) / 1e6, 1), "MB,", len(seeds), "seeds")
