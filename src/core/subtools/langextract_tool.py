
from typing import List, Dict, Any
from langextract import extract
from langextract.data import ExampleData, Extraction

class BasicLangExtractor:
    """
    
    """

    def __init__(
        self,
        # prompt: str,
        model_id: str = "gemini-1.5-flash",
    ):
        # self.prompt = prompt.strip()
        self.model_id = model_id             

    def extract_chunks(self, chunks: List[str],
                       examples: List[ExampleData] | None = None,
                       ) -> List[Dict[str, Any]]:
        """Run LangExtract on each text chunk and collect JSON outputs."""
        results: List[Dict[str, Any]] = []
        for chunk in chunks:
            out = extract(
                text_or_documents=chunk,                 # raw text
                # prompt_description=self.prompt,       
                examples=examples,                  # few-shot
                model_id=self.model_id,                  # LLM backend
            )
            # convert to ordinary Python dict so you can dump to JSONL
            results.append(out.to_json_dict())
        return results