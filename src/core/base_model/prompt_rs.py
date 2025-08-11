from typing import List, ClassVar, Literal
from pydantic import BaseModel, Field

class SubPrompt(BaseModel):
    """
    Defines a single, focused query for an internet search engine and the reasoning behind it.
    """
    query: str = Field(..., description="A concise and effective search query string.")
    reasoning: str = Field(..., description="A brief explanation of why this query is necessary to help answer the user's overall request.")


class ListPromptFm(BaseModel):
    """
    Given a user request, produce a JSON list of sub-prompts.
    """
    SYS_PROMPT: ClassVar[str] = (
        "You are an expert research assistant. Your task is to decompose an"
        "input into a series of {nofsub} queries "
        # "{rules}\n"
        "INPUT: {req}\n"
    )
    
    queries: List[SubPrompt] = Field(..., description="A list of targeted search prompts to be executed.")

    def to_str_queries(self):
        return [q.query for q in self.queries]