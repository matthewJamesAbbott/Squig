# Open WebUI Setup — DDC Book Library

The book index lives in Qdrant (collection `ddc-books`) with bge-m3
embeddings. Open WebUI is configured manually via its admin UI so it reuses
the *same* embedding model and reads the collection directly as an external
knowledge source.

> **Rule #1:** the embedding model Open WebUI uses for queries MUST be the
> same model used to build the index (`bge-m3` via Ollama at
> `http://localhost:11434`). Otherwise query vectors won't match the stored
> vectors and search quality collapses.

## 1. RAG embedding model

1. **Settings → Vector Storage → RAG → Embedding function**
2. Set the provider to **Ollama** and the model to **`bge-m3`**.
   - If Ollama is not on the default port/host (i.e. not
     `http://localhost:11434`), point the Ollama endpoint at your actual
     `EMBED_BASE_URL` instead.
3. Save.

This must match the model used to build the index (`EMBED_MODEL=bge-m3`).

## 2. External Knowledge Connection

1. **Admin → Knowledge → External → New Connection**
2. Provider: **Qdrant**
3. Endpoint: `http://localhost:6333`
4. API Key / auth: **none** (local-only Qdrant, no auth enabled)
5. Save / test the connection.

## 3. External Knowledge Base

1. **Admin → Knowledge → New → External**
2. Name: **`DDC Book Library`**
3. Connection: the Qdrant connection from step 2
4. Source: collection **`ddc-books`**
5. Content field: **`text`** (the page text payload field)
6. Metadata field: leave as the **entire payload** (so `book_title`,
   `authors`, `publisher`, `pub_date`, `ddc`, `ddc_coarse`, `ddc_tens`,
   `page_num`, `book_id` are all available as structured metadata in
   answers/citations).
7. Save.

## 4. Attach to a model

1. Open a chat, pick any LLM.
2. In the attachment / knowledge panel, attach the **`DDC Book Library`**
   knowledge base (it now appears as an external knowledge source).
3. Ask a question, e.g. *"how to ferment wine grapes"* — answers should be
   grounded in the book pages, with citations back to the payload metadata
   (book title, page number, DDC code).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| No results / bad results | Embedding model mismatch (step 1) — must be `bge-m3` |
| Connection test fails | Qdrant not running: `cd ~/DDC/indexer && docker compose up -d` |
| Knowledge base shows 0 documents | Wrong collection name or content field (must be `ddc-books` / `text`) |
| New books not appearing | Re-run the indexer (`python index_books.py --root ~/DDC`), then refresh the KB |
