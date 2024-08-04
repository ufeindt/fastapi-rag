from db import qdrant_client
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastui import AnyComponent, FastUI, prebuilt_html
from fastui.components import Heading, Markdown, ModelForm, Page
from fastui.forms import fastui_form
from llm import OpenAiMessage, openai_query, openai_query_async
from pydantic import BaseModel
from transformer import transformer
from typing_extensions import Annotated

app = FastAPI()


class QueryForm(BaseModel):
    query: str

    def get_messages(self, collection: str) -> list[OpenAiMessage]:
        query_embeddings = transformer.encode(self.query).tolist()
        results = qdrant_client.search(
            collection_name=collection,
            query_vector=query_embeddings,
            limit=20,
            with_payload=True,
        )
        context = "\n".join([result.payload["text"] for result in results])
        prompt = (
            "You are a helpful assistant that answers questions about D&D 5e. "
            "Given the following excerpts from the rules answer users questions. "
            "If you can't find the answer in the excerpts say you don't know. "
            f"Respond in markdown.\n{context}"
        )
        system_message = {"role": "system", "content": prompt}
        prompt_message = {"role": "user", "content": self.query}

        return [system_message, prompt_message]

    def submit_query(self, collection: str):
        return openai_query(
            model="gpt-3.5-turbo-1106",
            messages=self.get_messages(collection),
        )

    async def submit_query_async(self, collection: str):
        async for chunk in openai_query_async(
            model="gpt-3.5-turbo-1106",
            messages=self.get_messages(collection),
        ):
            yield chunk


@app.get("/api/", response_model=FastUI, response_model_exclude_none=True)
def homepage() -> list[AnyComponent]:
    return [
        Page(
            components=[
                Heading(text="FastAPI/FastUI RAG Demo", level=2),
                ModelForm(model=QueryForm, submit_url="/api/query/dnd-5e-srd"),
            ]
        )
    ]


@app.get("/api/collections")
async def collections():
    return {"collections": qdrant_client.get_collections()}


@app.post("/api/query/{collection}.json")
async def query_json(query_form: QueryForm, collection: str) -> StreamingResponse:
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")

    return StreamingResponse(
        query_form.submit_query_async(collection),
        media_type="text/event-stream",
    )


@app.post(
    "/api/query/{collection}", response_model=FastUI, response_model_exclude_none=True
)
async def query(
    query_form: Annotated[QueryForm, fastui_form(QueryForm)], collection: str
) -> list[AnyComponent]:
    return [Markdown(text=query_form.submit_query(collection))]


@app.get("/{path:path}")
def root() -> HTMLResponse:
    """Simple HTML page which serves the React app, comes last as it matches all paths."""
    return HTMLResponse(prebuilt_html(title="FastAPI/FastUI RAG Demo"))
