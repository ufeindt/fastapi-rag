from typing import Annotated, Union
from uuid import uuid4

from db import qdrant_client
from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llm import OpenAiMessage, openai_query, openai_query_async
from pydantic import BaseModel
from transformer import transformer

app_collections = FastAPI()
app_query = FastAPI()
app = FastAPI()

templates = Jinja2Templates(directory="templates")

app_collections.mount("/static", StaticFiles(directory="static"), name="static")
app_collections.add_middleware(GZipMiddleware, minimum_size=1000)

app.mount("/collections", app_collections)
app.mount("/query", app_query)

collections = {
    "dnd-5e-srd": {
        "slug": "dnd-5e-srd",
        "name": "D&D 5e SRD",
        "description": (
            "The basic rules for the 5th edition of Dungeons\xa0&\xa0Dragons from "
            "its System Reference Document (SRD)."
        ),
    }
}

queries = {}


class Query(BaseModel):
    query: str
    collection: str

    def get_messages(self) -> list[OpenAiMessage]:
        query_embeddings = transformer.encode(self.query).tolist()
        results = qdrant_client.search(
            collection_name=self.collection,
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

    def submit_query(self):
        return openai_query(
            model="gpt-3.5-turbo-1106",
            messages=self.get_messages(),
        )

    async def submit_query_async(self):
        async for chunk in openai_query_async(
            model="gpt-3.5-turbo-1106",
            messages=self.get_messages(),
        ):
            yield f"event: NextChunk\ndata: {chunk}\n\n"

        yield "event: ResponseComplete\ndata: End\n\n"


@app_collections.get("/", response_class=HTMLResponse)
async def get_collections(
    request: Request, hx_request: Annotated[Union[str, None], Header()] = None
):
    context = {
        "collections": collections.values(),
    }

    if hx_request:
        return templates.TemplateResponse(
            request=request,
            name="collections.html",
            context=context,
        )

    context["content_template"] = "collections.html"
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context=context,
    )


@app_collections.get("/{collection_name}", response_class=HTMLResponse)
async def get_collection(
    request: Request,
    collection_name: str,
    hx_request: Annotated[Union[str, None], Header()] = None,
):
    if collection_name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")

    context = {
        "collection": collections[collection_name],
    }

    if hx_request:
        return templates.TemplateResponse(
            request=request,
            name="collection.html",
            context=context,
        )

    context["content_template"] = "collection.html"
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context=context,
    )


@app_collections.post("/{collection_name}", response_class=HTMLResponse)
async def post_collection(
    request: Request, collection_name: str, query: Annotated[str, Form()]
):
    if collection_name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")

    query_id = str(uuid4())
    queries[query_id] = Query(query=query, collection=collection_name)
    return templates.TemplateResponse(
        request=request, name="query_sse.html", context={"query_id": query_id}
    )


@app_query.get("/{query_id}", response_class=StreamingResponse)
async def get_query(request: Request, query_id: str):
    if query_id not in queries:
        raise HTTPException(status_code=404, detail="Query not found")

    return StreamingResponse(
        queries[query_id].submit_query_async(),
        media_type="text/event-stream",
    )


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return RedirectResponse("/collections")
