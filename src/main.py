from typing import Annotated, Union
from uuid import uuid4

from db import qdrant_client
from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llm import OpenAiMessage, openai_query, openai_query_async
from pydantic import BaseModel
from transformer import transformer

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(GZipMiddleware)


class Todo:
    def __init__(self, text: str):
        self.id = uuid4()
        self.text = text
        self.done = False


todos = []


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="base.html")


@app.get("/todos", response_class=HTMLResponse)
async def list_todos(
    request: Request, hx_request: Annotated[Union[str, None], Header()] = None
):
    if hx_request:
        return templates.TemplateResponse(
            request=request, name="todos.html", context={"todos": todos}
        )
    return JSONResponse(content=jsonable_encoder(todos))


@app.post("/todos", response_class=HTMLResponse)
async def create_todo(request: Request, todo: Annotated[str, Form()]):
    todos.append(Todo(todo))
    return templates.TemplateResponse(
        request=request, name="todos.html", context={"todos": todos}
    )


@app.put("/todos/{todo_id}", response_class=HTMLResponse)
async def update_todo(request: Request, todo_id: str, text: Annotated[str, Form()]):
    for todo in todos:
        if str(todo.id) == todo_id:
            todo.text = text
            break
    return templates.TemplateResponse(
        request=request, name="todos.html", context={"todos": todos}
    )


@app.post("/todos/{todo_id}/toggle", response_class=HTMLResponse)
async def toggle_todo(request: Request, todo_id: str):
    for todo in todos:
        if str(todo.id) == todo_id:
            todo.done = not todo.done
            break
    return templates.TemplateResponse(
        request=request, name="todos.html", context={"todos": todos}
    )


@app.post("/todos/{todo_id}/delete", response_class=HTMLResponse)
async def delete_todo(request: Request, todo_id: str):
    for index, todo in enumerate(todos):
        if str(todo.id) == todo_id:
            del todos[index]
            break
    return templates.TemplateResponse(
        request=request, name="todos.html", context={"todos": todos}
    )


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


@app.get("/api/collections")
async def collections():
    return {"collections": qdrant_client.get_collections()}


@app.post("/api/query/{collection}", response_class=StreamingResponse)
async def query_json(query_form: QueryForm, collection: str) -> StreamingResponse:
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    return StreamingResponse(
        query_form.submit_query_async(collection),
        media_type="text/event-stream",
    )
