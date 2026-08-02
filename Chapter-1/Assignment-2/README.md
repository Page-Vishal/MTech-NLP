# Assignment 02 — Named Entity Recognition from a News Article

Identify and extract every **named entity** and its **entity type** from an English news
article using spaCy, export them to CSV, and summarise which types appear.

**Author:** Vishal Sigdel

## Files

| File | Role |
| --- | --- |
| `news.txt` | source article, plain text — "Broad Peak avalanche wipes out a generation of Nepali climbing greats" |
| `VishalSigdel_NER_01.ipynb` | the solution notebook |
| `VishalSigdel_NER_01.csv` | output — columns `Entity`, `Entity_Type` |
| [`Assignment_1.pdf`](../Assignment-1/Assignment_1.pdf) | assignment brief — one PDF covering both Assignment 01 and 02; the NER half starts on page 1 |

## Process

1. **Imports and the language model** — spaCy separates the library from its trained
   pipelines, so `en_core_web_sm` is installed once with
   `uv run python -m spacy download en_core_web_sm`. Its `ner` component is a
   transition-based model that labels *spans* of tokens, which is why `Nawang Thendu Sherpa`
   comes back as one entity rather than three.

2. **Load the article** — read `news.txt` with an explicit `utf-8` encoding inside a `with`
   block so the file closes even on error.

3. **Normalise the text** — same accent folding and curly-punctuation mapping as
   Assignment 01. Curly apostrophes make the tokenizer split possessives oddly, so
   `Pakistan's` can lose its entity boundary. Normalising runs *before* the pipeline, so the
   tokenizer and the NER model read the same string and character offsets stay aligned.

4. **Run the pipeline** — `nlp(text)` runs every component in `nlp.pipe_names`
   (`tok2vec`, `tagger`, `parser`, `ner`, …) and returns a `Doc`. Recognised entities land in
   `doc.ents` as `Span` objects, each carrying its text, its label, and its character offsets.

5. **Extract entities and types** — one row per `(entity.text, entity.label_)` pair.
   `strip()` is applied because a span can include trailing whitespace when the model ends an
   entity at a line break. `spacy.explain(label)` supplies each type's definition, so the
   summary table is self-documenting rather than relying on memorised tag names.

6. **Export** — build a `pandas.DataFrame` with the required `Entity` and `Entity_Type`
   columns and write it with `index=False` so pandas' row numbers stay out of the CSV.

7. **Summary of entity types** — three views: occurrences vs. distinct strings per type,
   the most-mentioned entities, and the full distinct list grouped by type.

8. **Bonus — visualisation** — `displacy.render(..., style="ent")` draws the first five
   sentences with each entity boxed and labelled in place, which makes mislabelled spans
   easy to spot by eye.

## Result

1451 tokens → **200** entity occurrences, **117** distinct entity strings across
**12** entity types.

| Type | Occurrences | Distinct | Meaning |
| --- | --- | --- | --- |
| `PERSON` | 49 | 28 | people, including fictional |
| `DATE` | 35 | 31 | absolute or relative dates or periods |
| `GPE` | 33 | 16 | countries, cities, states |
| `CARDINAL` | 23 | 13 | numerals not covered by another type |
| `ORG` | 20 | 15 | companies, agencies, institutions |
| `LOC` | 10 | 1 | non-GPE locations — mountain ranges, bodies of water |
| `NORP` | 10 | 6 | nationalities, religious or political groups |
| `ORDINAL` | 8 | 5 | "first", "second", etc. |
| `QUANTITY` | 7 | 4 | measurements of weight or distance |
| `LAW` | 2 | 2 | named documents made into laws |
| `PRODUCT` | 2 | 1 | objects, vehicles, foods |
| `TIME` | 1 | 1 | times smaller than a day |

Most-mentioned entity: `Everest` (`LOC`, 10 mentions).


## Notes

- **The CSV is one row per entity *occurrence*, not a unique entity list.** 200 occurrences
  reduce to 117 distinct strings. Deduping would lose the mention counts that show which
  names carry the story.

- **`en_core_web_sm` mislabels Himalayan proper nouns, and the output is reported as-is.**
  The model was trained on OntoNotes mostly US news and broadcast text so Nepali and
  Pakistani names sit outside its training distribution. Visible errors:

  | Entity | Predicted | Should be |
  | --- | --- | --- |
  | `Broad Peak` | `PERSON` | `LOC` |
  | `Nirmal Purja` | `ORG` (and `PERSON` elsewhere) | `PERSON` |
  | `Nepal` | `PERSON` (5×) | `GPE` |
  | `Kanchenjunga`, `Manaslu`, `Makalu`, `Annapurna` | `GPE` | `LOC` |
  | `Everest 10 times`, `Everest 15` | `LAW` | span error — should stop at `Everest` |
  | `Lhotse`, `Cholatse`, `Himlung Himal` | `PERSON` | `LOC` |

  Mountains landing in `GPE` rather than `LOC` is the dominant systematic error, which is
  why `LOC` has 10 occurrences but only **one** distinct string. The same spelling can also
  get two different labels in one article (`Nirmal Purja` as both `ORG` and `PERSON`) because
  the model decides per-context, not per-string.

- **Fixing this would need a larger or domain-adapted model.** `en_core_web_trf` (a
  transformer pipeline) or an `EntityRuler` seeded with a gazetteer of Himalayan peaks and
  Nepali surnames would correct most rows. `en_core_web_sm` was kept because it is CPU-only,
  a few megabytes, and adequate for demonstrating the pipeline — the trade-off is accuracy on
  exactly this article's vocabulary.

