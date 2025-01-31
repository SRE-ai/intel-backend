import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic import BaseModel
from typing import Optional
import os


class MergeConflictInput(BaseModel):
    conflict_text: str
    filename: str
    instructions: Optional[str] = None


class MergeConflictOutput(BaseModel):
    resolved_code: str


model = AnthropicModel(
    "claude-3-5-sonnet-latest",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


agent = Agent(
    model,
    system_prompt="You are an AI assistant skilled in using the Infinit API to get the latest changes.",
)


async def process_infinit_request(payload_request: dict) -> dict:
    """
    Helper function to process any infinit request with common headers and URL.
    """
    url = "https://sre.infinit.work/api/process"
    headers = {
        "accept": "application/json",
        "authorization": "947152230293129632542543215819.32639!eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZXNzaW9uIjoiMDAwZDMzNTk0ZWY2NDFiNDk2MjY0NTZmNTk1YTM4MGIiLCJlbWFpbCI6InN1cHBvcnRAaW5maW5pdC53b3JrIiwiaWF0IjoxNzM3OTIzNDc3LCJuYmYiOjE3Mzc5MjM0NzcsImV4cCI6MTczNzkyNzA3NywiYXVkIjoiZGVza3RvcC5pbmZpbml0LndvcmsiLCJpc3MiOiJpbmZpbml0LndvcmsifQ.bxGazQfmoAlDiH8baiRGGDWSRzzP4m7Fv4oG7i81i5k",
        "content-type": "application/json",
        "data-email": "support@infinit.work",
        "data-team": "SRE",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload_request)
        response.raise_for_status()  # Raises an error for non-200 responses
        return response.json()


@agent.tool
async def get_latest_changes(ctx: RunContext[MergeConflictInput]) -> dict:
    """
    This function sends a specific request to get the latest changes using the helper function.
    """
    payload = {
        "request": {
            "resource": "components",
            "resourceType": "sourceTracking",
            "mine": True,
            "range": "TODAY",
            "instance": "dev",
            "version": "62.0",
        },
        "user": "support@infinit.work",
        "team": "SRE",
        "logLevel": "SEVERE",
    }

    # Invoke the helper function with the payload
    return await process_infinit_request(payload)
