# RAG System v2 — Retrieval-Augmented Generation with a Chat LLM

A hands-on, self-contained tutorial notebook demonstrating a full Retrieval-Augmented
Generation (RAG) pipeline, built to run on Google Colab (also works locally/CPU).

This is a rebuilt, robust version of an earlier notebook. The original relied on
`pipeline(...)[0]["generated_text"]`, which breaks across `transformers` versions since
the output key changed. This version fixes that by:

1. Using a different generator model — **`Qwen/Qwen2.5-0.5B-Instruct`**, a small modern
   instruction-tuned chat LLM — instead of FLAN-T5.
2. Calling the model with `tokenizer` + `model.generate()` **directly** instead of the
   high-level `pipeline`, so there is no dependency on any output dictionary key. Tokens
   are decoded manually, so it works on any recent `transformers` release.

## What is RAG?

A plain LLM answers only from its frozen training weights, which causes hallucination
(confident wrong answers) and staleness (no knowledge of new or private data). RAG fixes
this by:

1. **Retrieving** relevant passages from a knowledge base via semantic vector search
2. **Augmenting** the prompt with those passages
3. **Generating** an answer grounded in them — no fine-tuning required

```
 User Query -> Embed Query -> Vector Search (FAISS) -> top-k chunks
                                                          |
                                                          v
                                   Augmented Prompt (query + context)
                                                          |
                                                          v
                                        Chat LLM (Qwen) -> Final Answer
```

## What's inside the notebook

`RAG_Systems_v2_ChatLLM.ipynb` walks through the full pipeline end to end:

1. **Environment setup** — installs `transformers`, `sentence-transformers`, `faiss-cpu`,
   `accelerate`
2. **Knowledge base** — a small set of 12 short articles about Transformers, attention,
   BERT/GPT, embeddings, RAG, vector search, chunking, hallucination, and tokenization
3. **Chunking** — splits documents into overlapping word-window chunks (`chunk_size=60`,
   `overlap=15`) so information near a boundary isn't lost
4. **Embeddings** — encodes chunks with `all-MiniLM-L6-v2` (384-dim), L2-normalized for
   cosine similarity via inner product
5. **Vector store** — builds a FAISS `IndexFlatIP` index over the chunk embeddings
6. **Retrieval** — embeds the query and returns the top-k most similar chunks
7. **Generator** — loads `Qwen/Qwen2.5-0.5B-Instruct` and calls `model.generate()`
   directly (via a chat template), avoiding the fragile `pipeline` output-key dependency
8. **Prompt augmentation** — injects retrieved chunks into the prompt and instructs the
   model to answer only from that context
9. **`RAGSystem` class** — a reusable pipeline combining the embedder, FAISS index, and
   chat LLM behind `retrieve()` and `answer()` methods
10. **RAG vs. no-RAG comparison** — same question answered with and without retrieved
    context, to see how grounding changes the answer
11. **Retrieval evaluation** — Precision@k over a small labeled test set of queries
12. **Exercises** for extending the knowledge base and tuning chunking parameters

## Getting started

### Option 1: Google Colab (recommended)

Open the notebook in Colab and run the cells top to bottom. For faster generation, switch
to a GPU runtime via `Runtime -> Change runtime type -> T4 GPU` (CPU also works, just
slower).

### Option 2: Local environment

```bash
pip install -r requirements.txt
jupyter notebook RAG_Systems_v2_ChatLLM.ipynb
```

A CUDA-capable GPU is optional but speeds up generation; the code auto-detects
`cuda`/`cpu`.

## Tech stack

| Library | Role |
|---|---|
| `sentence-transformers` | Text -> dense embeddings (the **R**etriever) |
| `faiss-cpu` | Fast nearest-neighbor vector search |
| `transformers` | Loads the chat LLM that generates the answer (the **G**enerator) |
| `torch` | Model inference backend |
| `accelerate` | Efficient model loading/device placement |

**Models used:**
- Embedder: `all-MiniLM-L6-v2`
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`

## Example result

Precision@3 retrieval accuracy across 6 test queries: **83.3%**

```
Q: What is cosine similarity used for?
A: Cosine similarity is used to measure the similarity between two vectors by calculating
   their cosine of the angle between them. This metric ignores the magnitude of the
   vectors, making it suitable for comparing text embeddings.
Sources used:
  - Cosine Similarity (score=0.771)
  - Vector Databases and Similarity Search (score=0.336)
  - Word and Sentence Embeddings (score=0.320)
```

## Exercises

1. Add 3 new documents to `KNOWLEDGE_BASE` and re-run — confirm they get retrieved.
2. Change `chunk_size` and `overlap` and observe the effect on Precision@3.
3. Ask a question whose answer is **not** in the knowledge base. Does the model correctly
   say it lacks information?
