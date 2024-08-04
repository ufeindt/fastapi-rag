import os
from typing import TypedDict

from openai import AsyncOpenAI, OpenAI

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
openai_client_async = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class OpenAiMessage(TypedDict):
    role: str
    content: str


def openai_query(model: str, messages: list[OpenAiMessage]):
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


async def openai_query_async(model: str, messages: list[OpenAiMessage]):
    stream = await openai_client_async.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        yield chunk.choices[0].delta.content or ""
