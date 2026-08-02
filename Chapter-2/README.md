# Chapter 2 — Statistical NLP

Foundations of statistical NLP: frequency counts, vocabulary, co-occurrence,
sparsity, smoothing, bigram prediction, TF-IDF, vector similarity, and
conditional probability — building up from word counting to a bigram
language model.

**Author:** Vishal Sigdel

## Files

| File | Role |
| --- | --- |
| `Exercise.pdf` | assignment brief — 11 exercises plus a mini-pipeline assignment |
| `VishalSigdel_StatisticalNLP_01.ipynb` | Exercises 1–11, worked in one notebook |
| `VishalSigdel_StatisticalNLP_02.ipynb` | standalone pipeline for Exercise 11 — interactive bigram next-word predictor |

## `_01`: Exercises 1–11

Each exercise states the objective, then works through it in code, then prints
the result:

1. **Word Frequency Counter** — tally words in a sentence into a dict (TF).
2. **Vocabulary Builder** — unique words across a corpus via `set`.
3. **Word Co-occurrence Matrix** — window-size-1 neighbour counts.
4. **Detect Data Sparsity** — list zero-count pairs in the co-occurrence matrix.
5. **Laplace Smoothing** — add-one smoothed unigram probabilities.
6. **Simple Keyboard Prediction** — bigram counts, top-3 next word by raw frequency.
7. **TF-IDF** — TF, DF, IDF and TF-IDF over a 3-document corpus.
8. **Dense Vector Similarity** — cosine similarity between two embeddings.
9. **Find Similar Words** — nearest embedding by cosine similarity.
10. **Mini NLP Pipeline (assignment)** — combines 1–6 over a new corpus: vocabulary,
    frequency, TF, co-occurrence, sparsity, Laplace-smoothed unigrams, bigram
    prediction.
11. **Conditional Probability Prediction** — bigram counts → $P(w_2 \mid w_1) =
    \frac{Count(w_1, w_2)}{Count(w_1)}$ → ranked next-word probabilities → top-3.

## `_02`: Conditional Probability Pipeline (standalone)

Pulls Exercise 11 out into its own pipeline notebook, driven by real user input
instead of a hardcoded query word:

1. Load the training corpus.
2. Build bigram counts.
3. Compute conditional probabilities for every bigram.
4. Prompt the user for a word (`input()`).
5. Display every possible next word with its conditional probability.
6. Predict the single highest-probability next word.
7. **Bonus** — show the top 3 predictions, ranked.

The prompt ("Enter a word: ") is printed as its own cell output *before* the
`input()` cell, rather than passed as `input()`'s prompt argument — some
notebook front ends (VS Code's Jupyter extension included) don't render an
`input()` prompt string inside the inline input box, leaving it blank. Printing
it separately guarantees it's visible above the box.

## Notes

- Corpora are intentionally tiny (4–5 short sentences) so every count and
  probability can be checked by hand.
- `_02`'s bigram model has no smoothing — a query word never seen as the first
  half of a bigram returns no candidates.
