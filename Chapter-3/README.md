# Chapter 3 — Word Embeddings

Three programming assignments on distributional word representations, each a self-contained
notebook. Small corpora throughout, so every notebook runs end to end on a laptop CPU in minutes
rather than hours.

Rough runtimes on one CPU core: **A3 ~10 min** (the from-scratch training loop dominates; set
`epochs=1` in its `Config` for a two-minute smoke run), **A4 ~2 min** after GloVe is downloaded,
**A5 well under a minute** after BERT is cached. First run adds the downloads in the table below.

| # | Notebook | Objective |
|---|---|---|
| 3 | [Assignment-3_Word2Vec_SGNS.ipynb](Assignment-3_Word2Vec_SGNS.ipynb) | Implement Skip-Gram with negative sampling from scratch in NumPy; evaluate on similarity and analogies |
| 4 | [Assignment-4_Pretrained_Embeddings.ipynb](Assignment-4_Pretrained_Embeddings.ipynb) | Use pre-trained GloVe in a sentiment classifier; compare performance with and without embeddings |
| 5 | [Assignment-5_Contextual_vs_Static.ipynb](Assignment-5_Contextual_vs_Static.ipynb) | Compare BERT contextual embeddings against static GloVe/SGNS on word-sense disambiguation and sentence similarity |

Run them in order. Assignment 4 loads the SGNS vectors that Assignment 3 saves, and Assignment 5
uses both those and the GloVe file that Assignment 4 downloads. Each dependency degrades gracefully
with a message rather than an exception if the earlier notebook has not been run.

## Corpora and models

| Resource | Size | Used by | Source |
|---|---|---|---|
| NLTK `brown` | 988 K tokens after normalisation | A3 | `nltk.download("brown")` |
| NLTK `movie_reviews` | 2 000 reviews | A4 | `nltk.download("movie_reviews")` |
| GloVe 6B.100d | 822 MB zip → 347 MB | A4, A5 | downloaded from nlp.stanford.edu |
| `bert-base-uncased` | 440 MB | A5 | Hugging Face hub |

NLTK corpora download automatically inside the notebooks. GloVe and BERT download on first run into
`data/` and the Hugging Face cache respectively.

## What each notebook covers

### Assignment 3 — Word2Vec from scratch

Pure NumPy, no `gensim`, no autograd. Sections follow the pipeline: tokenisation policy →
vocabulary indexing → context windows → the negative-sampling objective → frequent-word subsampling
→ hand-derived gradients (verified against finite differences) → SGD training loop → evaluation.

Evaluation is four probes: nearest neighbours, Spearman correlation against a WordSim-353 subset,
analogies, and an odd-one-out / heatmap / PCA look at the structure. Measured on this corpus:
$\rho \approx 0.35$ against human similarity judgements, odd-one-out solved on every group, and
analogies that split cleanly — morphological and syntactic ones resolve, purely semantic ones
(`man : woman :: king : ?`) do not. The notebook works through why frequency, not the algorithm,
decides which relations survive.

### Assignment 4 — Pre-trained embeddings downstream

Sentiment classification on the Pang & Lee polarity dataset. Five models on one split, holding the
architecture fixed so any difference is attributable to the embedding table:

1. TF-IDF + logistic regression (no embeddings)
2. Mean GloVe + logistic regression
3. Neural bag-of-words, random init, learned
4. Neural bag-of-words, GloVe init, frozen
5. Neural bag-of-words, GloVe init, fine-tuned

Plus §7, which swaps in the Assignment-3 SGNS vectors — same architecture, same dimensionality, only
the pre-training corpus differs (1 M tokens of Brown vs 6 B of Wikipedia), and coverage differs with
it: GloVe supplies vectors for 98.4% of the task vocabulary against our SGNS's 41.5%.

Measured outcome: **fine-tuned GloVe wins overall at 0.857** test accuracy, just above TF-IDF's 0.850
and clearly above random init's 0.813. The **frozen** arm comes last of the three (0.703) — the
notebook works through why that is a trainable-capacity limit (it underfits, with a negative
train-val gap and its best epoch last) rather than a verdict on GloVe, since the fine-tuned arm uses
the very same vectors.

### Assignment 5 — Contextual vs static

BERT as a frozen feature extractor via Hugging Face `AutoModel`, all 13 hidden-state layers
extracted. Two analyses:

* **Word-sense disambiguation** — 5 polysemous words × 2 senses × 2 sentences. GloVe's
  same-sense-minus-cross-sense margin is exactly 0 by construction; BERT reaches **+0.31** at layer
  10, with 20/20 same-sense retrieval.
* **Sentence similarity** — paraphrase vs same-topic-opposite-meaning vs unrelated pairs, chosen so
  that lexical overlap points the wrong way. Every mean-pooled representation fails this, BERT
  included — the notebook shows why pooling, not the vectors, is the culprit, and what SBERT does
  about it.

Then interpretability: anisotropy measurements, GloVe neighbour/arithmetic inspection (possible
because the table is a finite object), and watching one word's BERT vector move across contexts
(impossible for a static table).

## Outputs

Written to `artifacts/`, git-ignored:

| File | Written by | Contents |
|---|---|---|
| `sgns_embeddings.txt` | A3 | trained vectors in word2vec text format |
| `sgns_model.npz` | A3 | `E`, `U`, vocabulary, counts |
| `sgns_config.json` | A3 | the run's hyperparameters |
| `assignment4_results.json` | A4 | test accuracy per model, GloVe coverage |
| `assignment5_results.json` | A5 | per-layer WSD margins, sentence-similarity gaps |

## Environment

Dependencies are declared in the repository `pyproject.toml`; `uv sync` installs them. Beyond the
Chapter 1–2 set, Chapter 3 adds `torch`, `transformers`, `scikit-learn`, and `datasets`.
