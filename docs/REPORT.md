# Tvek: detecting how certain a Swedish speaker sounds

Joakim Gustafson (KTH Speech, Music and Hearing) with Claude Code. Status 2026-11-22. Model: CertNet v7 of the KTH svtx toolkit (Tvek 1.2;
Tvek 1.0–1.1 shipped CertNet v3, still available with `model="v3"`).

## Abstract

Tvek labels each utterance of a Swedish speaker as **certain**, **uncertain** or **no claim** (a question, backchannel or greeting)
and gives a running curve over the words. It listens to the voice: frozen VoxRex speech representations and eGeMAPS prosody, per word,
normalised per speaker, through a small causal transformer, averaged with a cue model on utterance-end cues. It was trained on 12,500
utterances of Swedish face-to-face dialogue with soft targets and 600 utterances tagged by ear in four rounds (two of them
chosen by active learning).
On held-out recording sessions it separates certain from uncertain utterances at **AUC 0.83** (cross-validated by session), against 0.62
for text-only LLM judgements; on radio interviews, a domain it never saw, at **AUC 0.85** (random sample 0.91). The three-way label is
weaker (50 % correct on interviews), and the labels themselves are noisy: one listener repeated 8 of 15 earlier labels.

**Version 1.2 (CertNet v7)** adds a third speech style, a three-party estimation quiz game recorded with a far-field microphone array
(Swedish and English), with 300 utterances tagged by ear and weak labels from the dialogue structure for the other 5,100 utterances. In
cross-validation with one recording held out per fold it is the best model on the face-to-face dialogue (AUC 0.817, 0.822 with the cue
model), close to the interview-trained baseline on the interviews (0.848 with the cue model against 0.855) and equal to it on the quiz
game (0.696); a variant trained on the quiz tags only is better on the quiz game (0.721; English 0.60 against 0.54) and is used there
(section 6). Nothing besides the voice helped on the quiz game: dialogue structure, cue words, laughter, facial expression, gaze aversion
and creak added to CertNet did not raise the cross-validated AUC.

## 1. The task

*Speaker certainty* here is **perceived** certainty: how sure the speaker sounds to a listener, the "feeling of another's knowing" (Brennan
& Williams, 1995), not whether they are right or how sure they feel (the two diverge; Pon-Barry & Shieber, 2011). It is judged per
utterance (a stretch of one speaker's speech between pauses of 0.8 s or more), with three classes, because many utterances make no
claim at all. A curve over the words shows where uncertainty sets in, for interactive systems that must react while the person speaks.

## 2. Related work

**Cues to uncertainty in speech.** In question answering, people who are less sure answer later, more often with fillers ("eh"), rising
intonation and hedges (Smith & Clark, 1993); listeners use exactly these cues to judge another's certainty (Brennan & Williams, 1995).
Visual cues (marked facial expressions, gaze aversion) add to the audio and are strongest with it (Swerts & Krahmer, 2005). In Swedish,
certainty is also coded by modal particles: *ju* claims shared, certain knowledge, *väl* asks for confirmation, *kanske* and *nog*
weaken a claim, and stranded function words before a pause (*och…*, *att…*) mark hesitation (see the reference list of the research
report behind this work, *Swedish Voice Uncertainty Detection*, 2026).

**Automatic detection.** Liscombe, Hirschberg & Venditti (2005) detected certainness in spoken tutorial dialogue from prosody and
lexical features; Forbes-Riley & Litman (2011) built a tutor that detects and adapts to student uncertainty in real time; Pon-Barry &
Shieber (2011) recognised uncertainty in speech with prosodic features computed separately for the target word and its context; Dral,
Heylen & op den Akker (2011) separated uncertain from neutral utterances with prosodic features alone. Text-only certainty models exist
for English scientific writing (Pei & Jurgens, 2021) and hedges, and a recent semi-supervised speech-confidence detector fused a
Whisper encoder with eGeMAPS (arXiv 2605.12387). **We found no published certainty or uncertainty detector for spoken Swedish**, and no
public Swedish data labelled for it, so no published system can be compared on our test sets; the English results above use other
labels, corpora and speakers.

**Building blocks.** VoxRex (Malmsten, Haffenden & Börjeson, 2022), a Swedish wav2vec 2.0 model trained on Swedish radio; eGeMAPS
(Eyben et al., 2016) from openSMILE (Eyben, Wöllmer & Schuller, 2010); word-level audio-text fusion after CogniAlign (GitHub
davidorp/CogniAlign); Monte Carlo dropout (Gal & Ghahramani, 2016) and BALD (Houlsby et al., 2011) to choose what to tag.

## 3. Data

| Data | What | Size | Use |
| --- | --- | --- | --- |
| face-to-face dialogue corpus | Swedish dialogue, close-talk microphones, one channel per speaker | 12,500 utterances (all speakers' utterances of 3–40 words chosen for certainty labelling) | training (soft targets) |
| Tags by ear, rounds 1–4 | one listener, certain / uncertain / no claim | 600 utterances (430 certain or uncertain) | training (weight 20) and cross-validated evaluation |
| Tags by ear, round 5 | the same listener, Swedish radio interview programmes and a radio discussion programme | 120 utterances (73 certain or uncertain) + 15 hidden repeats | test (v3); training and cross-validation (v6, v7) |
| Three-party estimation quiz game | 5 games of about 55 min, 3 players + a game master on a loudspeaker, 2013 lab recordings; 2 games Swedish, 3 English; one beam per player from a 7-microphone array (far field, other players audible) | 5,423 player utterances | weak labels (v7, weight 1) |
| Tags by ear, rounds 6–7 | the same listener, quiz game, by ear from the player's beam (+ 4 s of room context on demand), transcript hidden unless asked | 300 utterances (238 certain or uncertain: 170 uncertain) | training (weight 20) and cross-validation (v7) |

**Soft targets** for the untagged utterances mix (a) the votes of three text LLMs (Qwen 27B, Gemma 31B, ChatGPT) given the utterance
and its dialogue context, (b) a cue model (multinomial logistic regression on 14 utterance-end cues: last word a function word or a
filler, voicing that trails past the last word, silence after, duration, speech rate, end F0 slope, F0 spread, lengthening of the last
word, the soft tail's level, hedges and boosters), fitted on the training folds' tags only, and (c) in the final model, the answers of
an audio-language model fine-tuned on the tags. **Tagging rounds 3–4** were chosen by active learning: the 100 utterances where the three
seeds and their MC-dropout samples disagreed most (BALD) and the 50 where the model and the text votes disagreed most, at most 3 per
session.

## 4. Model

Per word, plus a TAIL token for the 1.2 s after the last word (cut at the speaker's next word):

- **audio:** VoxRex (`KBLab/wav2vec2-large-voxrex-swedish`) hidden layers 3, 6, 9, 12, 15, 18, averaged over the word and over its last
  100 ms, mixed by learned softmax weights, projected to 256;
- **prosody:** eGeMAPS v02 frame features averaged per word, F0 and loudness standard deviation, voiced share, duration, lengthening
  (log seconds per letter relative to the utterance), F0 slope, the pause after, position from the end, and flags for function words,
  fillers, hedges and boosters; z-scored per speaker over all the speaker's utterances, projected to 256;
- the two projections added, plus a position embedding; a 2-layer causal transformer (width 256, 4 heads; token *i* sees words up to
  *i*); a 3-way head on every token. The TAIL token gives the utterance answer, the others the curve.

Loss: cross-entropy of the TAIL token against the utterance target plus the earlier tokens towards the same target, weighted 0.3 ×
(position / n)². AdamW, learning rate 3·10⁻⁴, one-cycle schedule, 8 epochs, batch 64, dropout 0.2; three seeds averaged. The released
label averages the network's TAIL probabilities with the cue model's.

## 5. Results

**Cross-validation on the dialogue corpus** (5 folds by recording session, AUC certain vs uncertain on the tags):

| System | Tags | AUC |
| --- | --- | --- |
| text-LLM votes (Qwen, Gemma, ChatGPT) | 239 | 0.62 |
| audio-language model fine-tuned on the tags | 240 | 0.61 |
| cue model (logistic regression on 14 end cues) | 239 | 0.73 |
| CertNet v1, audio + prosody | 239 | 0.73 (0.75 with the cue model) |
| CertNet v2 | 335 | 0.76 (0.77) |
| **CertNet v3 (Tvek)** | **430** | **0.81 (0.83)** |
| cue model alone, same folds | 430 | 0.78 |

**What did not help** (same setting, all 430 tags): adding the text stream (KB-BERT with 30 s of context: 0.77); turn-end features and a
voice-activity-projection hold probability as inputs or extra targets (0.81–0.82); other audio encoders, alone or added to VoxRex:
self-supervised VoxRex 0.79, KB-Whisper-large encoder 0.80, audEERING's dimensional emotion model 0.75, VoxRex + its arousal / dominance
/ valence 0.81, VoxRex + self-supervised VoxRex 0.815, VoxRex + KB-Whisper 0.79. Each tagging round added about 0.02, the encoders
nothing: the labels limit the model.

**Radio interviews** (never trained on; 120 utterances, half where the model was least sure and half at random):

| Test | Certain / uncertain | AUC (95 % bootstrap interval) |
| --- | --- | --- |
| all | 42 / 31 | **0.85** (0.75–0.93) |
| random half | 22 / 17 | **0.91** (0.79–0.99) |

| Model ↓ / listener → | certain | uncertain | no claim | can't tell |
| --- | --- | --- | --- | --- |
| certain | 30 | 13 | 24 | 2 |
| uncertain | 3 | 14 | 6 | 0 |
| no claim | 9 | 4 | 14 | 1 |

Three-way agreement is 50 %. The largest confusion is the interviewer's long, statement-like question, which the model hears as a
certain statement and the listener as no claim.

**Label reliability.** 15 earlier face-to-face-dialogue utterances were shown again among the interview items: the listener gave the same label 8
times; seven moved to *uncertain* (three from *certain*, four from *no claim*). Listener agreement on voice characteristics is modest
in general (Wiechmann et al., 2023, found novice agreement around κ = 0.2 that a short explanation did not raise), and these numbers
bound what any model can reach on such labels.

## 6. Version 1.2: CertNet v7 (2026-11-20/21)

**Why another domain.** The quiz game was designed to elicit uncertainty: three players must agree on numerical estimates (the area of
the world's 8th largest country, a year, a speed) within about three minutes and then give them to a game master. It is task-oriented,
multi-party, partly English and recorded far-field, so it tests how far a model trained on face-to-face Swedish dialogue carries over,
and it adds labelled data of a kind the dialogue corpus has little of (proposals, estimates, agreements).

**Features.** The same features as at inference (`CertNet.features`), computed on each player's own microphone-array beam; the prosody
z-scored over all of that player's utterances (as for the interviews). Utterances: a player's words (KB-Whisper with fillers for
Swedish, CrisperWhisper for English) split at pauses of 0.8 s or more.

**Tags.** Rounds 6 and 7, 150 utterances each, 30 per game, utterances of three words or more. Round 6: per game 12 the weak labels
called uncertain, 12 certain, 6 no claim, half of each where the weak-label sources disagreed most, half at random. Round 7, by active
learning on the then current model: 12 most likely certain (certain utterances are rare), 10 nearest the decision boundary, 4 where the
model and the dialogue structure or cue words disagreed, 4 at random. The listener tagged 170 of the 238 claims uncertain: in this game
even agreements and the final answers are mostly said uncertainly. The round-7 picks the model thought most likely certain were tagged
certain 30 % of the time against 13 % for the others.

**Weak labels** (for the untagged quiz utterances): a log-odds sum of hand-set contributions: game phase and position in the
discussion; proposals of a value (initial, revised, final, and whether it was the team's eventual answer), agreements, giving the answer
and its correctness (from a local LLM's reading of each round: the question, every proposal, agreements, the team's answer and the
correct answer read by the game master, converted to numbers; correct = within a factor of 10^0.1); fillers, hedges and boosters;
CertNet v6 on the Swedish games only; frown events (Py-Feat AU4); eye-tracker gaze aversion; laughter right after the utterance. Only
these hand-set weights are used as training targets: weights refitted on the tags would carry tag information across the folds.

**Cross-validation.** 5 folds: the dialogue-corpus sessions as for v3, every 5th interview recording, and one quiz game per fold, so
every number below comes from recordings the model never trained on; 3 seeds averaged; AUC certain vs uncertain on the tags.

| Configuration (same folds) | Dialogue (430) | Interviews (73) | Quiz game (238) | Quiz, Swedish | Quiz, English |
| --- | --- | --- | --- | --- | --- |
| v6: the v3 recipe + the interview tags | 0.799 (0.815) | **0.807 (0.855)** | 0.696 | 0.748 | 0.535 |
| v7_tags: + the quiz tags (weight 20) | 0.812 (0.817) | 0.776 (0.845) | **0.721** | **0.752** | **0.604** |
| **v7 = v7_weak1: + the quiz tags + the weak labels (weight 1)** | **0.817 (0.822)** | 0.793 (0.848) | 0.696 | 0.746 | 0.552 |

(In parentheses: averaged with the cue model, as Tvek outputs.) The quiz tags help on the quiz game, almost all of it on English; the
weak labels help the dialogue corpus a little and cancel the quiz gain. **Tvek 1.2 ships v7_weak1**, the best on Swedish two-person
dialogue and the closest to v6 elsewhere; v7_tags is used for the quiz game itself. A first round with 150 tags and weak labels refitted
on those tags had shown 0.73–0.74 on the quiz game: the refit leaked tag information across the folds, and with clean weak labels the
number fell to the table's.

**What did not help on the quiz game** (238 tagged claims, CertNet's out-of-fold score + one group, logistic regression cross-validated by
game; 95 % bootstrap interval of the difference): the position in the discussion +0.005 (−0.02..+0.03); creak (Knarr word detector:
the last word's score, the fillers' maximum) +0.008 (−0.02..+0.04); laughter −0.013; facial expression and gaze (AU4, brow raise,
measured gaze aversion) −0.012; the dialogue structure −0.05; everything together −0.09 (−0.15..−0.02). With the weights refitted on
the 150 first tags, the structural and lexical cues even took unexpected signs (hedges counted towards certain): in this game nearly
everything is uncertain, so cues of uncertainty separate little.

**Creak.** On all 38,482 player words (Knarr detector; 15 % scored creaky): a creaky last word and more creaky words go with
*certain* utterances (AUC 0.40–0.42 for uncertain, i.e. less creak there), as phrase-final creak marks a finished statement; creaky
fillers go with *uncertain* (0.61; Swedish 0.79 on 55 utterances). CertNet already carries most of it.

**Faces on the dialogue corpus.** On 90 tagged dialogue utterances with synced camera video (19 sessions), facial features (MediaPipe
blendshapes, Py-Feat action units, head motion; 84 features) did not help as a whole (0.50–0.62 alone, no gain with CertNet), but
eye-in-head gaze aversion during the utterance alone reached AUC 0.68–0.73, independent of CertNet (r = −0.01): CertNet + gaze aversion
0.77 against 0.69 in the same cross-validation (difference against CertNet's raw score +0.04, 95 % interval −0.04..+0.12). Promising,
not yet shown; it needs more tagged utterances with video.

## 6.1 Uncertainty and creak (2026-11-21/22)

Creak is often heard in hesitant speech, so CertNet v7 was used to choose words for a round of creak labelling by ear (the creak detector
Knarr's round 3): from face-to-face dialogue, radio interviews, the quiz game and podcasts, 236,491 scored words, sampled as the last
words of the most uncertain utterances (45), of the most certain ones (45), fillers in uncertain utterances (35), utterance-medial words of
uncertain utterances (25), random words (25) and 15 hidden repeats; at most 3 words per person. One listener labelled the 190 words on a
page with the waveform and glottal pulses, spectrogram, pitch and period plots and half-speed playback (58 creaky, 127 not, 5 unsure; the
15 repeats got the same label).

| Group | Creaky |
| --- | --- |
| fillers in uncertain utterances | 51 % |
| last word of an uncertain utterance | 33 % |
| last word of a certain utterance | 28 % |
| medial word of an uncertain utterance | 28 % |
| random word | 4 % |

Creak is far more common at utterance ends and in fillers than elsewhere, but at utterance ends CertNet's p(uncertain) does not separate
creaky from modal words (AUC 0.49): an uncertain utterance does not end in creak more often than a certain one. This agrees with section 6,
where creak scores added to CertNet did not help, and with the creak detector's later finding that creak at phrase ends is predicted by
the prosody leading into the word and by who is speaking, not by an estimated lung state (Knarr's docs/REPORT.md). Creak is therefore not
used as a certainty cue.

## 7. Limitations

- One listener's labels; the model learns that listener's percept, and the labels are noisy (above).
- Trained on face-to-face dialogue (close-talk), radio interviews and a far-field quiz game; other speech styles are untested. English
  is supported only as far as the quiz game's English tags go (AUC 0.55–0.60): Tvek is a Swedish model.
- The prosody is normalised per speaker over the utterances given, so short inputs (fewer than ~10 utterances) are unreliable.
- Word times are needed; the demo's KB-Whisper word times are rough.
- Questions and statements with question function are the weakest point of the three-way label.

## 8. Reproduction

Training code and the full experiment record are in the KTH svtx repository (`research/certainty/`, TTS dialogue report sections 37,
42–44 and 44.9; technical report section 7.1). CertNet v7: `cert_gg_feats.py` (quiz-game features), `cert_net7.py cv` (the table of
section 6) and `cert_net7.py final v7_weak1`; the quiz-game data, tags, weak labels and the round extraction are in the quiz corpus
repository. The package here contains the inference code, the weights (three seeds + cue model, hash-checked; `MANIFEST.json` = v7,
`MANIFEST_certnet_v3.json` = v3) and `tools/export_weights.py`, which renames the later trainer's stream-named keys (`a.vox.*` →
`a.*`); exported v7 weights reproduce the trainer's probabilities to 1e-7.

## References

- Brennan, S. E., & Williams, M. (1995). The feeling of another's knowing: Prosody and filled pauses as cues to listeners about the
  metacognitive states of speakers. *Journal of Memory and Language*, 34(3), 383–398.
- Dral, J., Heylen, D., & op den Akker, R. (2011). Detecting uncertainty in spoken dialogues: an exploratory research for the automatic
  detection of speaker uncertainty by using prosodic markers. In *Affective Computing and Sentiment Analysis*, Springer.
  [link](https://research.utwente.nl/en/publications/detecting-uncertainty-in-spoken-dialogues-an-explorative-research/)
- Eyben, F., Wöllmer, M., & Schuller, B. (2010). openSMILE: the Munich versatile and fast open-source audio feature extractor. *ACM
  Multimedia 2010*.
- Eyben, F., et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and affective computing. *IEEE
  Transactions on Affective Computing*, 7(2), 190–202.
- Forbes-Riley, K., & Litman, D. (2011). Benefits and challenges of real-time uncertainty detection and adaptation in a spoken dialogue
  computer tutor. *Speech Communication*, 53(9–10).
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML 2016*.
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning.
  arXiv:1112.5745.
- Liscombe, J., Hirschberg, J., & Venditti, J. J. (2005). Detecting certainness in spoken tutorial dialogues. *Interspeech 2005*.
- Malmsten, M., Haffenden, C., & Börjeson, L. (2022). Hearing voices at the National Library: a speech corpus and acoustic model for the
  Swedish language. arXiv:2205.03026.
- Pei, J., & Jurgens, D. (2021). Measuring sentence-level and aspect-level (un)certainty in science communications. *EMNLP 2021*.
- Pon-Barry, H., & Shieber, S. M. (2011). Recognizing uncertainty in speech. *EURASIP Journal on Advances in Signal Processing*, 2011.
- Smith, V. L., & Clark, H. H. (1993). On the course of answering questions. *Journal of Memory and Language*, 32(1), 25–38.
- Cheong, J. H., Jolly, E., Xie, T., Byrne, S., Kenney, M., & Chang, L. J. (2023). Py-Feat: Python facial expression analysis
  toolbox. *Affective Science*, 4, 781–796.
- Wiechmann, J., Rautenberg, F., Wagner, P., & Häb-Umbach, R. (2023). Explaining voice characteristics to novice voice practitioners:
  how successful is it? *ICPhS 2023*, 4224–4228.
- Swerts, M., & Krahmer, E. (2005). Audiovisual prosody and feeling of knowing. *Journal of Memory and Language*, 53(1), 81–94.
- A semi-supervised framework for speech confidence detection (2026). arXiv:2605.12387.
  [link](https://arxiv.org/html/2605.12387v1)
- CogniAlign: [github.com/davidorp/CogniAlign](https://github.com/davidorp/CogniAlign).
