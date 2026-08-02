# Assignment 01 — POS Extraction from a News Article

Extract every **noun** and **verb** from an English news article using NLTK and export them to CSV.

**Author:** Vishal Sigdel

## Files

| File | Role |
| --- | --- |
| `news.txt` | source article, plain text — "Broad Peak avalanche wipes out a generation of Nepali climbing greats" |
| `VishalSigdel_POS_01.ipynb` | the solution notebook |
| `VishalSigdel_POS_01.csv` | output — columns `Word`, `POS_Tag` |
| `Assignment_1.pdf` | assignment brief |

## Process

1. **Imports and NLTK resources** — NLTK ships models separately from the library, so `punkt`, `punkt_tab` (tokenizers) and `averaged_perceptron_tagger*` (POS model) are downloaded once into `~/nltk_data`.

2. **Load the article** — read `news.txt` with an explicit `utf-8` encoding inside a `with` block so the file closes even on error.

3. **Normalise the text** — news sites use smart typography. Two cleanup steps keep the tokenizer from producing noisy tokens:
   - accent folding via `NFKD` decomposition, then dropping combining marks (`café` → `cafe`)
   - mapping curly quotes, en/em dashes and non-breaking spaces to ASCII, so `don’t` and `don't` are not treated as different tokens

4. **Tokenize** — `word_tokenize` splits on whitespace *and* linguistic boundaries: punctuation becomes its own token and contractions split (`don't` → `do`, `n't`). The tagger takes this token list as input.

5. **POS tagging** — `nltk.pos_tag` returns `(word, tag)` pairs in the **Penn Treebank** tagset. Tags are context-sensitive, so the same spelling can tag differently depending on its neighbours (`climbing` is `VBG` in "climbing greats", `NN` elsewhere).

6. **Filter to nouns and verbs** — Penn Treebank splits each class by inflection, so the notebook matches against explicit tag sets:
   - nouns: `NN`, `NNS`, `NNP`, `NNPS`
   - verbs: `VB`, `VBD`, `VBG`, `VBN`, `VBP`, `VBZ`

   A small `classify(tag)` helper maps a tag to `"Noun"` / `"Verb"` / `None`, keeping the filter readable and easy to extend.

7. **Export** — build a `pandas.DataFrame` with the required `Word` and `POS_Tag` columns and write it with `index=False` so pandas' row numbers stay out of the CSV.

8. **Summary** — total count plus a per-tag breakdown from `value_counts()`.

## Result

1372 tokens → **633** extracted words: 457 nouns, 176 verbs.

## Run it

```bash
uv run --with jupyter --with nltk --with pandas \
  jupyter nbconvert --to notebook --execute --inplace VishalSigdel_POS_01.ipynb
```

Or open the notebook in Jupyter / VS Code and run all cells. Working directory must be
`Assignment-1/` — the paths `news.txt` and `VishalSigdel_POS_01.csv` are relative.

## Notes

- Duplicate words appear multiple times: the CSV is one row per *token occurrence*, not a
  vocabulary list. Deduping would lose frequency information.
- The Averaged Perceptron tagger is statistical, so a few tags will be wrong on unusual
  phrasing — expected, and cheaper than a transformer model for this task.
