from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
import os
from pydantic import BaseModel
from typing import Optional


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
    deps_type=MergeConflictInput,
    result_type=MergeConflictOutput,
    system_prompt="You are an AI assistant skilled in resolving Git merge conflicts.",
)

internal_agent = Agent(
    OpenAIModel(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
)


@agent.tool
async def resolve_merge_conflict(
    ctx: RunContext[MergeConflictInput], data: MergeConflictInput
) -> MergeConflictOutput:
    """
    Resolves a merge conflict using Claude Sonnet 3.5.
    """
    prompt = f"""
    You are resolving a Git merge conflict in '{data.filename or "a file"}'. Ensure that the final resolution 
    is clean and maintains all necessary changes. If instructions are provided, follow them.

    Merge Conflict:
    ```
    {data.conflict_text}
    ```

    Instructions:
    {data.instructions or "None provided"}

    Provide only the resolved code.
    """
    async with internal_agent.run_stream(prompt) as response:
        # Resolved code
        return await response.get_data()
