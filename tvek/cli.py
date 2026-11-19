"""tvek AUDIO WORDS [--channel N] [--tier NAME] [--gap 0.8] [--device cuda|mps|cpu] [-o out.tsv] [--json out.json]

WORDS: the words of ONE speaker, as a Praat TextGrid (the first interval tier, or --tier NAME) or a TSV/CSV with a header naming
start, end and word (or without a header: start, end, word). Writes TSV: start, end, label, p_certain, p_uncertain, p_no_claim, text;
--json adds the per-word curve and the cues."""
import argparse
import csv
import json
import sys


def read_words(path, tier=None):
    if str(path).lower().endswith(".textgrid"):
        import re
        txt = open(path, encoding="utf-8", errors="replace").read()
        for t in re.split(r'item \[\d+\]:', txt)[1:]:
            name = re.search(r'name = "(.*?)"', t).group(1)
            if 'IntervalTier' not in t or (tier and name != tier): continue
            iv = re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', t, re.S)
            return [(w.strip(), float(a), float(b)) for a, b, w in iv if w.strip()]
        sys.exit(f"no interval tier {tier or ''} in {path}")
    rows = [r for r in csv.reader(open(path, encoding="utf-8"), delimiter="\t" if str(path).endswith(".tsv") else ",") if r]
    head = [c.strip().lower() for c in rows[0]] if rows else []
    if "start" in head and "end" in head:
        i, j = head.index("start"), head.index("end"); k = head.index("word") if "word" in head else head.index("label")
        return [(r[k], float(r[i]), float(r[j])) for r in rows[1:]]
    out = []
    for r in rows:
        try: out.append((r[2], float(r[0]), float(r[1])))
        except (ValueError, IndexError): continue
    return out


def main():
    ap = argparse.ArgumentParser(description="Tvek: how certain a Swedish speaker sounds, per utterance")
    ap.add_argument("audio"); ap.add_argument("words"); ap.add_argument("--channel", type=int); ap.add_argument("--tier")
    ap.add_argument("--gap", type=float, default=0.8); ap.add_argument("--device"); ap.add_argument("-o", "--out"); ap.add_argument("--json")
    a = ap.parse_args()
    from . import CertaintyDetector
    res = CertaintyDetector(a.device).score_file(a.audio, read_words(a.words, a.tier), channel=a.channel, gap=a.gap)
    f = open(a.out, "w", newline="") if a.out else sys.stdout; w = csv.writer(f, delimiter="\t")
    w.writerow(["start", "end", "label", "p_certain", "p_uncertain", "p_no_claim", "text"])
    for r in res: w.writerow([f'{r["start"]:.3f}', f'{r["end"]:.3f}', r["label"], r["p"]["certain"], r["p"]["uncertain"], r["p"]["no_claim"], r["text"]])
    if a.json: json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
