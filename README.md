# Tvek

**How certain a Swedish speaker sounds**, per utterance and word by word. Tvek labels each utterance **certain**, **uncertain**
(hesitant, unsure, guessing, trailing off) or **no claim** (a question, a backchannel, a greeting), and gives a running curve over the
words, so you can see where the doubt sets in. It listens to the voice, not only the words: on Swedish dialogue sessions it has not
been trained on, it separates certain from uncertain utterances at **AUC 0.83** (cross-validated by session), where text-only LLM
judgements of the same kind of utterances reached 0.62. *Tvek* is Swedish for hesitation.

```bash
pip install "tvek @ git+https://github.com/jockegustafson/Tvek.git"
tvek interview.wav words.tsv --channel 0 -o certainty.tsv     # words of ONE speaker: TSV (word, start, end) or a Praat TextGrid
```

```
start    end      label      p_certain  p_uncertain  p_no_claim  text
```

One row per utterance (the words split at pauses of 0.8 s or more, and stretches over 40 words or 20 s at their longest pause);
`--json` adds the curve per word and the cues behind the label.

## Run it from Python

```python
from tvek import CertaintyDetector

det = CertaintyDetector()                                      # GPU if there is one, Apple silicon (mps), else CPU
words = [("jag", 12.31, 12.40), ("tror", 12.40, 12.62), ...]   # (word, start s, end s) of one speaker, e.g. from a forced aligner
for u in det.score_file("interview.wav", words, channel=0):
    print(u["start"], u["end"], u["label"], u["p"], u["text"])
    # u["curve"]: [P(certain), P(uncertain), P(no claim)] after each word
```

- **One speaker per call, all their words at once.** Prosody is normalised over the speaker's utterances in the call (about 10 or more
  are needed), as a listener calibrates to a voice. For a stereo dialogue, score each channel with its own words.
- **Word times** come from a forced aligner (MFA, WhisperX, a Praat TextGrid); the words themselves matter for a few cues (hedges such
  as *kanske*, *nog*, *jag tror*, final fillers), but the model mainly listens.
- The curve is causal: the value after word *i* is computed from words 1..*i* only, so it can be read while the person speaks.

## How it works

Per word, and for a TAIL token covering the 1.2 s after the last word (where "men…" or "och eh…" trail off), from the speaker's own
channel:
- **audio:** VoxRex (`KBLab/wav2vec2-large-voxrex-swedish`), layers 3, 6, 9, 12, 15, 18, averaged over the word and over its last
  100 ms, mixed by learned weights;
- **prosody:** eGeMAPS (openSMILE) per word, F0 and loudness spread, voicing, duration, lengthening, F0 slope, the pause after,
  position, and flags for function words, fillers, hedges and boosters, z-scored per speaker.

A 2-layer causal transformer turns these into the three classes at every word (three seeds averaged); the utterance label averages
the TAIL token with a small cue model (logistic regression on end cues such as a final function word, a final filler, voice trailing
past the last word, the final pitch slope). The model is **CertNet v3** of the KTH svtx toolkit.

## Training and evaluation

- **Data:** 12,500 utterances of Swedish face-to-face dialogue (lab recordings, KTH) with soft targets from text-LLM votes, a cue model and an
  audio-LLM teacher, plus 600 utterances tagged by ear in four rounds (rounds 3–4 chosen by active learning), weighted 20×.
- **Evaluation:** 5-fold cross-validation by recording session, AUC certain vs uncertain on the 430 tagged certain / uncertain items:

| Model | AUC |
| --- | --- |
| text-LLM votes (Qwen, Gemma, ChatGPT), on the first 239 tags | 0.62 |
| cue model alone | 0.78 |
| CertNet (audio + prosody) | 0.81 |
| **Tvek = CertNet + cue model** | **0.83** |

Each tagging round added about 0.02; other audio encoders (self-supervised VoxRex, the KB-Whisper encoder, emotion dimensions) did not
improve it.

**Radio interviews** (never trained on): 120 utterances tagged by ear from Swedish radio interview programmes
and a radio discussion programme, half where the model was least sure and half at random (2026-11-19):

| Interview test | Items (certain / uncertain) | AUC certain vs uncertain (95 % bootstrap interval) |
| --- | --- | --- |
| all 120 | 73 (42 / 31) | **0.85** (0.75–0.93) |
| the random half | 39 (22 / 17) | 0.91 (0.79–0.99) |

Separating certain from uncertain carries over to interviews. The three-way label is weaker (50 % correct): the model often calls an
interviewer's long, statement-like question *certain* where the listener heard *no claim* (24 of 120). **Limits:** one listener's
tags; on 15 earlier face-to-face-dialogue utterances shown again without saying so, the same listener gave the same label 8 times, so the labels
themselves are noisy, especially between *no claim* and *uncertain*. Treat single labels as estimates and look at the probabilities.

## Licence

Non-commercial research use only (see `LICENSE`). Please cite this repository. Questions: Joakim Gustafson, KTH Speech, Music and
Hearing.
