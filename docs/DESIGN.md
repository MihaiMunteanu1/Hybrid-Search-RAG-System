# Design decisions

Why the system is built the way it is, with the measurement behind each choice. Numbers
marked *(reference)* were measured in the earlier prototype of this project (`RAGPipeling`)
and carried over; the rest were measured on this corpus. Decisions were made on the
development set only (25 questions, 22 answerable). The test sets were run once each.

## Documents and chunking

| Decision | Reason |
|---|---|
| Every format is converted to markdown, with headings as `#` lines | One section parser then covers every format, and chunks can follow the author's structure. |
| PDF headings come from, in order: the outline (bookmarks), font size and weight, then wording (`Capitolul II`, `Art. 5`) | Most PDFs have no outline, and some are set in a single font. In an EUR-Lex act, the font heuristic found 1 heading in 88 pages because chapters and articles are set in the body font. The wording tier fixes that. |
| Wording headings must be numbered in increasing order | A correlation table at the end of NIS2 ("Article 1", "Article 2" alone on their lines) otherwise added 27 false headings. |
| Running headers and footers are removed (repeated text in the page margins, same font size) | Otherwise they would be read as a heading on every page and repeated in every chunk. |
| Soft hyphens at line breaks are joined before invisible characters are removed | Removing them first split "interna-țional" into two words, 68 times in GDPR. |
| Cedilla letters (ş ţ) are normalized to comma-below (ș ț) | BM25 would otherwise treat the two spellings as different words. |
| Spreadsheets drop empty columns and the empty tail of each row | In a form-like spreadsheet, half of every chunk was pipes and spaces. Those near-empty vectors reached the dense top 5 for 8 of the 25 development questions. |
| PDF tables become rows "\| a \| b \|", with the category or the column name repeated in each row; only tables with at most 70% empty cells and short cells (added after the evaluation) | As plain text a table put every cell on its own line, and a value lost its label. Measured on two sets of table questions only, so the other results are untouched: retrieval 14/18 → 17/18 and 10/12 → 11/12 (the second set written before the last change); correct answers 12 → 13 and 6 → 7. The remaining errors are the 4B model misreading rows ("x" taken as "not applicable"), not extraction. Boxed paragraphs, charts and forms are rejected and stay text; PDFs without such tables come out byte-identical. |
| A web page keeps only its main content: no menus, link lists or URLs (added after the evaluation) | The saved regulation page became 54 chunks of site menu and news that matched almost any question; it is now 8. On test v2 (seen, so a confirmation) correct answers went from 33/40 to 34/40 and unanswerable questions refused from 3/4 to 4/4: without the menu entries about fees, the model refused cleanly instead of writing a refusal in prose. |
| Chunks of up to 400 tokens with a 60-token overlap, counted with the embedding model's tokenizer | The embedding model reads at most 512 tokens and silently drops the rest. 400 leaves room for the heading path and the prefix. |
| `structure` strategy: one chunk per section; small sections merged; long ones split by paragraph, then sentence | A chunk then matches a unit the author wrote. Before the merging rules, heading-only sections became 40 chunks of a few tokens, containing no facts. |
| `structure` kept after comparing it with `fixed` and `semantic` | With the reranker the three are within one question of each other on dev (R@5: 20, 19, 20 of 22; MRR 0.81, 0.80, 0.78), so recall does not decide. Without the reranker, fixed windows are clearly worse at rank 1 (7 vs 11–13 of 22). `semantic` took ~31 minutes to chunk on a CPU; `fixed` chunks are ~40% larger, so the model reads more per answer. `structure` follows the sections, so citations carry a readable heading path. Per-question results in `data/eval/results/dev-chunking-*.json`. |

## Indexing and search

| Decision | Reason |
|---|---|
| Embedding model `intfloat/multilingual-e5-base`, with the `query:` / `passage:` prefixes | Multilingual (Romanian and English). `e5-large` was measured and rejected: with the reranker, 19/22 on dev against 20/22 for `e5-base`, with indexing about 3 times slower. |
| The chunk's heading path is prepended to the text that is embedded and indexed by BM25 | A short passage often does not name its own topic. Recall@5 rose by 6.9 points: 76% → 83% *(reference)*. |
| BM25 with Romanian and English Snowball stemming, applied to every token, and accents folded | Romanian is heavily inflected. Without stemming, 0 of 13 inflected word pairs matched; with it, 12 of 13 *(reference)*. Folding accents makes questions typed without diacritics match. |
| Hybrid search fused with RRF (k = 60) | Dense and BM25 fail on different questions. Dev recall@5: dense 16/22, BM25 15/22, hybrid 18/22. Test v1: 26/41, 22/41, 25/41. |
| RRF ties broken by the dense rank, then by chunk id | Exact ties are common. Breaking them by chunk id (a hash) made recall@1 move by 10 points when only a folder name changed. |
| Search per language: one dense and one BM25 list per corpus language | A Romanian question otherwise favours Romanian chunks, and an English chunk that answers it falls below the cut. |
| Cross-encoder reranker (`mmarco-mMiniLMv2-L12-H384-v1`) over 20 fused candidates, top 5 kept, 512-token window | The largest single improvement (table below). With a 256-token window, a fact at token 296 scored −7.63; with 512 it scored −1.93 *(reference)*. |
| No refusal threshold on the reranker score | The top-1 scores of answerable and unanswerable questions overlap. On dev, one unanswerable question scored +0.48, above the lowest answerable one (−1.10). |

Retrieval configurations compared on the development set (22 answerable questions):

| Configuration | recall@5 | MRR | Time per question |
|---|---|---|---|
| Hybrid, no reranker | 18/22 | 0.617 | ~0.1 s |
| + reranker, one list, k = 10, 20 candidates | 19/22 | 0.758 | ~2 s |
| **+ reranker, per language, k = 5, 20 candidates** (final) | **20/22** | **0.777** | ~2 s |
| + reranker, per language, k = 10, 40 candidates | 19/22 | 0.773 | ~3.3 s |
| + reranker, per language, k = 25, 100 candidates | 19/22 | 0.764 | ~7 s |

The differences are one or two questions out of 22, so this is a choice, not a proof.
More candidates did not help and cost seconds on a CPU.

## Generation

| Decision | Reason |
|---|---|
| Qwen3.5-4B (Q4_K_M GGUF), served by llama.cpp in a separate process | See the comparison below. A separate server avoids compiling Python bindings on Windows, and the code stays the same if the server moves to a GPU machine. |
| Temperature 0 | At 0.1, the same prompt came back with citations in 2 of 4 runs and without them in the other 2, so a single run of the evaluation set could not be reproduced. |
| "Thinking" mode disabled | It would cost minutes on a CPU, and the facts are already in the prompt. |
| Sources numbered `[1]`…`[5]`, each with its document, heading path and page | The model can tell similar passages apart by chapter, and each citation shown to the user points somewhere readable. |
| Refusal through an exact marker, `NOT_FOUND` | Detecting refusals with a regex over prose misread 2 correct refusals *(reference)*. |
| A reply whose first sentence says the sources lack the answer, citing nothing yet, is a refusal (added after the evaluation) | The model sometimes writes a refusal in prose and cites sources after it. On all 165 stored answers this rule caught 7 such replies, all judged wrong or missed, and no correct or partial answer. A negative answer ("Nu există taxă [2].") cites its source in the first sentence, so it is not caught. |
| Prompt rules 4 and 5: no refusals written in prose; answer only the part the sources cover | Refusals passing as answers fell from 3 of 10 to 0 of 11, with no answerable question lost *(reference)*. |
| A reply with no citation gets one follow-up request for citations | At temperature 0 the model often answers correctly but cites nothing. A first wording of the request turned 3 of 3 refusals into "answers" with citations; the current wording does not. The follow-up reuses llama-server's prompt cache, so it costs only the new turn. |
| The answer language is named explicitly ("Answer in Romanian"), detected from the question | When told to "answer in the language of the question", a candidate model answered two Romanian questions in English. The detector got 19 of 22 questions right, 0 wrong and 3 undecided *(reference)*; an undecided question falls back to the generic instruction. |
| llama-server threads: 6 for writing, 14 for reading the prompt (i9-13900H, set in `.env`) | Measured with llama-bench: 6 threads read at 55 tok/s and write at 10.1 tok/s; 14 threads read at 66 tok/s and write at 9.4 tok/s. Writing is memory-bound and the efficiency cores slow it down. |

Model comparison on 5 questions from this corpus (CPU):

| Model | Correct | Cited | In the question's language | s / answer |
|---|---|---|---|---|
| Llama-3.2-3B | 4/5 | 5/5 | 4/5 | 7.0 |
| Gemma-3-4B | 5/5 | 5/5 | 3/5 | 8.8 |
| **Qwen3.5-4B** | **5/5** | **5/5** | **5/5** | 13.3 |

Llama answered one question with nothing but "[2]". Gemma answered two Romanian questions in
English. Five questions can rule a model out but cannot prove it is the best one; the
evaluation sets confirm the choice.

## Evaluation

| Decision | Reason |
|---|---|
| Relevance is defined by a gold quote: a retrieved chunk is relevant if it overlaps an occurrence of the quote | This works for any chunking strategy and survives changes to the loaders, because the quotes are located again on every run. |
| Wilson 95% confidence intervals for every rate | With about 20–40 questions, one question is 2–5 points. Wilson stays valid near 0% and 100%. |
| Answer correctness is judged by hand, not by an LLM judge | An automatic judge misses partially correct answers *(reference)*. |
| A development set for every decision, and new test sets run once | A set that has been tuned on is optimistic. Recall@5 without the reranker was 82% on dev and 61% on test v1. |

## Web app

| Decision | Reason |
|---|---|
| The answer streamed as server-sent events: sources, then text, then the parsed answer | The sources are ready in about 2 s and the answer takes about a minute on a CPU. The final answer may differ from the streamed text (a refusal, or a rewrite with citations). |
| One answer generated at a time | On a CPU, two generations in parallel each run at half speed. |
| The generator is closed explicitly when the client disconnects (`ClosingStream`) | Starlette leaves it suspended inside the lock, and every later question waited for it. |
| A disconnect cancels the model's request at once: `ClosingStream` watches for it and closes the connection to llama-server, which reads prompts in steps of 256 tokens | Starlette notices a client that left only at the next token, so a question abandoned while the model read its sources still ran to the end. With the default step of 2048, llama-server also kept reading for ~21 s after a cancel. Measured: the next question's first token came after 47.8 s instead of 85.5 s; reading speed was unchanged (55.6 tokens/s with either step). |
| `index.html` is served with `Cache-Control: no-cache` | After a rebuild, a cached page pointed at deleted script files. The scripts carry a hash in their names and stay cacheable. |
| Uploads are written under a temporary name and moved into place only after they were read and chunked | An unreadable file never replaces the version already indexed. |
