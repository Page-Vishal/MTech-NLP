# MTech-NLP

Coursework repository for the MTech NLP module, one folder per assignment, each a
self-contained Jupyter notebook with its input data, its output, and its own README.

**Author:** Vishal Sigdel

## Contents

| Folder | What it offers |
| --- | --- |
| [Chapter-1/Assignment-1/](Chapter-1/Assignment-1/) | Part-of-speech extraction. Tags an English news article with NLTK and exports every noun and verb to CSV. |
| [Chapter-1/Assignment-2/](Chapter-1/Assignment-2/) | Named entity recognition. Finds the people, places, organisations and dates in the same article with spaCy and exports them to CSV. |
| [Chapter-2/](Chapter-2/) | Statistical NLP. 11 exercises (frequency, vocabulary, co-occurrence, smoothing, TF-IDF, cosine similarity, conditional probability) plus a standalone interactive bigram next-word predictor. |

Top-level files:

| File | Role |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Python version and shared dependencies for every assignment |
| [uv.lock](uv.lock) | pinned dependency versions — reproducible installs |

## Setup

Managed with [uv](https://docs.astral.sh/uv/). Python 3.14+.

```bash
uv sync
```

Assignment-2 additionally needs spaCy's English pipeline, which ships separately from the
library:

```bash
uv run python -m spacy download en_core_web_sm
```

## Running a notebook

Notebooks use **relative paths**, so the working directory must be the assignment folder.

```bash
cd Chapter-1/Assignment-1
uv run --with jupyter jupyter nbconvert \
  --to notebook --execute --inplace VishalSigdel_POS_01.ipynb
```

Or open the folder in Jupyter / VS Code and run all cells.