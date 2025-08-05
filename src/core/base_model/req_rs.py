from __future__ import annotations
from typing import ClassVar, Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator

AreaLevel = Literal["global", "local"]

def clarify_requirement(req: RequirementFm, max_rounds=2):
    while req.get_questions() and max_rounds > 0:
        max_rounds -= 1
        answers_dict = {}

        for q in req.get_questions():
            answer = input(f"{q}\nYour answer: ")
            field = next(k for k, v in req._QUESTIONS.items() if v == q)
            answers_dict[field] = answer            
        req.apply_answers(answers_dict)
    

class RequirementFm(BaseModel):
   
   
    time_start: Optional[str] = Field(None, description="Start timestamp or date")
    time_end: Optional[str] = Field(None, description="End timestamp or date")

    
    area_value: Optional[str] = Field(None, description="Explicit area name")
    
    entity_type: Optional[str] = Field(None, description="Domain or entity type (e.g. real-estate)")

    scope: AreaLevel = Field("local", description="Global research or local research")

    additional_details: Optional[List[str]] = Field(
        None,
        description="Any extra clarifications or details"
    )
    topic: Optional[str] = Field(
        "natural",
        description="A specific sub-area or focus (e.g. housing prices, land usage)"
    )
    model_config = ConfigDict(extra='forbid')  


    @model_validator(mode="after")
    def _absorb_extras(cls, values: "RequirementFm") -> "RequirementFm":
        """
        Any unknown keys (because extra='allow') end up as attributes on the instance.
        We move them into additional_details, keeping the model tidy.
        """
        known = set(cls.model_fields.keys())
        for key in list(values.__dict__.keys()):
            if key not in known:
                values.additional_details[key] = values.__dict__.pop(key)
        return values

    _QUESTIONS: ClassVar[Dict[str,str]] = {
        "time_start": "Provide the start of the time span.",
        "time_end": "Provide the end of the time span.",
        "area_value": "Provide the explicit area name (e.g. 'Ho Chi Minh City').",
    }

    def get_questions(self) -> List[str]:
        #TODO: Add more questions based on the current state
        """
        Return the list of clarification questions for missing data.
        This also sets needs_clarification & clarification_questions.
        """
        clarification_questions = []
        for field, question in self._QUESTIONS.items():
            value = getattr(self, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                clarification_questions.append(question)
        
        return clarification_questions
        
    
    def apply_answers(self, answers: Dict[str, str | None]) -> "RequirementFm":
        """
        answers example: {"time_end": "2025", "area_value": "Ho Chi Minh City"}
        Returns a *new* RequirementFm with those fields patched and questions recalculated.
        """
        # print(type(self.model_copy(update=answers, deep=True)))
        self = self.model_copy()
        for field, value in answers.items():
            if value is not None:
                setattr(self, field, value)
        
    
    def is_complete(self) -> bool:
        return not self.get_questions()

