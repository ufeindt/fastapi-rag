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
        "name": "D&D 5th Edition System Reference Document",
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
            # for chunk in self.query.split(" "):
            yield f"event: NextChunk\ndata: {chunk}\n\n"
            # await sleep(0.1)

        yield "event: ResponseComplete\ndata: End\n\n"


@app_collections.get("/", response_class=HTMLResponse)
async def get_collections(
    request: Request, hx_request: Annotated[Union[str, None], Header()] = None
):
    if hx_request:
        return templates.TemplateResponse(
            request=request,
            name="collections_block.html",
            context={"collections": collections},
        )

    return templates.TemplateResponse(
        request=request, name="collections.html", context={"collections": collections}
    )


@app_collections.get("/{collection}", response_class=HTMLResponse)
async def get_collection(
    request: Request,
    collection: str,
    hx_request: Annotated[Union[str, None], Header()] = None,
):
    if hx_request:
        return templates.TemplateResponse(
            request=request,
            name="collection_block.html",
            context={"collection": collection},
        )
    return templates.TemplateResponse(
        request=request, name="collection.html", context={"collection": collection}
    )


@app_collections.post("/{collection}", response_class=HTMLResponse)
async def post_collection(
    request: Request, collection: str, query: Annotated[str, Form()]
):
    if collection not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")

    query_id = str(uuid4())
    queries[query_id] = Query(query=query, collection=collection)
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
