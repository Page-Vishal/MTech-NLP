# Word Embeddings from Scratch — Skip-gram with Negative Sampling

Pure-NumPy implementation of SGNS (Mikolov et al., 2013). No `gensim`, no autograd —
every gradient is derived and written out by hand.


Then *Run All*. The first cell downloads the **text8** corpus (31 MB zipped → 100 MB of
Wikipedia text, ~17 M tokens) into `corpus.txt`. To use a different corpus, place your own
plain-text `corpus.txt` in this folder before running — the download is skipped if the file
already exists.

## Notebook structure

The notebook follows the assigned 8-step procedure section by section:

| § | Step | Key content |
|---|---|---|
| 1 | Assemble corpus | text8 download, tokenisation policy, streaming reader |
| 2 | Vocabulary size $M$ | frequency counting, `min_count` + top-$M$ filters, Zipf plot |
| 3 | Context window $C$ | $2C+1$ sliding window, dynamic-window distance weighting |
| 4 | Co-occurrence dictionary | `(target, context) → count` over a corpus sample |
| 5 | Embedding size $N$ | memory/quality trade-off |
| 6 | Initialise $E$ and $U$ | why two tables; noise and subsampling distributions |
| 7 | Train | gradient derivation, finite-difference check, vectorised SGD loop |
| 8 | Keep $E$, discard $U$ | cosine query interface, save/load |

Followed by evaluation (nearest neighbours, analogies, odd-one-out, similarity heatmap,
PCA projection) and a discussion of design choices and limitations.

## Beyond the stated procedure

Two additions from the original paper, both with measurable effect on quality:

* **Frequent-word subsampling** ($t = 10^{-4}$) — without it, `the`/`of`/`and` dominate the
  training pairs and content words barely train.
* **Unigram$^{0.75}$ noise distribution** — uniform negatives almost never sample a frequent
  word; raw-frequency negatives sample almost nothing else.

The co-occurrence dictionary (Step 4) is built for inspection only. SGNS streams pairs directly
from the corpus and never needs the full table — which is precisely why it scales past the
count-based methods it replaced.

## Configuration

All hyperparameters live in one `Config` dataclass:

| Parameter | Default | Meaning |
|---|---|---|
| `vocab_size` | 50,000 | $M$ — top-$M$ word types kept |
| `min_count` | 5 | drop words rarer than this |
| `window` | 2 | $C$ — 2 left + 2 right, a 5-word window |
| `dim` | 100 | $N$ — embedding dimensionality |
| `negatives` | 5 | $k$ — negative samples per positive pair |
| `epochs` | 5 | passes over the corpus |
| `lr` | 0.025 | initial learning rate, linearly decayed |
| `subsample_t` | 1e-4 | frequent-word discard threshold |
| `ns_power` | 0.75 | noise-distribution exponent |

Runtime is roughly 20–40 minutes on a laptop CPU at these defaults. For a fast check, set
`vocab_size=5000, dim=50, epochs=1` — that runs in a couple of minutes and still produces
recognisable neighbours.

## Outputs

Written to `artifacts/` (git-ignored):

| File | Contents |
|---|---|
| `embeddings.txt` | $E$ in word2vec text format — loadable by `gensim.KeyedVectors` |
| `model.npz` | $E$, $U$ and the vocabulary, for resuming training |
| `config.json` | the exact hyperparameters of the run |

## References

1. Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their
   Compositionality.* NeurIPS.
2. Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR.
3. Goldberg & Levy (2014). *word2vec Explained.* arXiv:1402.3722.
4. Levy & Goldberg (2014). *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS.
