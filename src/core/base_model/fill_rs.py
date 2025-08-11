
from typing import List, ClassVar, Type, Any, Dict
from pydantic import BaseModel, Field
from langextract.data import ExampleData, Extraction as LEExtraction   


class Extraction(BaseModel):
    label: str = Field(..., description="Semantic category (e.g. organization, product, date)")
    value: str = Field(..., description="Surface string extracted from the text")



class ExampleDataFm(BaseModel):
    """
    Lightweight wrapper you use inside your codebase.
    The `to_langextract()` method (our add-on hook) converts it to
    langextract.data.ExampleData on demand.
    """

    SYS_PROMPT: ClassVar[str] = """
    You are “Example-Data Generator”, an assistant that fabricates *single-paragraph* news-style snippets and returns them as a JSON object that conforms to the Pydantic schema `ExampleDataFm`.
    ◆  Required behaviour
    1. You will receive a list of feature labels in the runtime variable {req} (e.g. ["organization", "product", "event"]).
    2. Write one coherent, original English paragraph (≈40-80 words) that *mentions each* label exactly once in given context: {topic}.
    3. Produce an `extractions` array where:
        • Each object’s **label** is one of the labels in {req} (case-sensitive).  
        • **value** is the exact surface string that appears in the paragraph for that label.  
        • The ordering of objects must follow the order of {req}.
    4. Do **NOT** include any additional labels that are not in {req}.
    5. Output must be **valid JSON only** – no markdown, no code fences, no trailing commas.  
    Example:
    {{
    "text": "TechCorp launched its latest drone, the SkyZoom X, during a major unveiling event in Hanoi.",
    "extractions": [
        {{ "label": "organization", "value": "TechCorp" }},
        {{ "label": "product", "value": "SkyZoom X" }},
        {{ "label": "event", "value": "unveiling event" }}
    ]
    }}

    ◆  Validation tips
    - Make sure every `value` string appears verbatim in `text`.
    - Keep `text` factual-sounding but entirely fictional to avoid copyright.
    - Do not include newline characters inside JSON strings.
    Return the JSON object and nothing else.
    """

    text: str
    extractions: List[Extraction]


def to_langextract(
          *,
          _input: ExampleDataFm
          ) -> ExampleData:
        """
        Convert self into a langextract.data.ExampleData instance.
        """
        return ExampleData(
            text=_input.text,
            extractions=[
                LEExtraction(ext.label, ext.value)
                for ext in _input.extractions
            ],
        )