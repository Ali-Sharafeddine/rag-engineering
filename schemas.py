from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class Source(BaseModel):
    text: str
    score: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]