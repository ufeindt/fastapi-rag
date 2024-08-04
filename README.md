# FastAPI/FastUI RAG

A web api and interface for asking questions about documents. The project uses 
[retrieval-augmented generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation):
- The documents, e.g., the Dungeons and Dragons 5th edition Systems Reference
  Document (DnD 5e SRD), is split into chunks, which are converted into
  embeddings.
- The embeddings and text chunks are then stored in a vector DB (in this case,
  [Qdrant](https://qdrant.tech/)).
- In the web interface, the user enters a question that is also converted into
  an embedding. Then the 20 most relevant chunks are retrieved from the vector
  database.
- Lastly the question is sent to a [large language model
  (LLM)](https://en.wikipedia.org/wiki/Large_language_model), using the chunks
  as context, and the LLM response is displayed in the interface.

## Acknowledgements

This project is in large parts based on [this demo notebook](https://github.com/crspeller/dnd-answers) by [Christopher Speller](https://github.com/crspeller).

## Current Stack
- Embeddings model:
  [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v10)
- Vector DB: [Qdrant](https://qdrant.tech/) (used in local mode for now)
- Web API: [FastAPI](https://fastapi.tiangolo.com/)
- Web UI: [FastUI](https://github.com/pydantic/FastUI)
- LLM: GPT-3.5-turbo-1106 (for now because it is cheaper; requires an OpenAI API key to
  use)

## To Do 
- Implement the UI 
