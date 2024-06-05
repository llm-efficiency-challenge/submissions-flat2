from pydantic import BaseModel

from typing import List, Dict, Optional


class ProcessRequest(BaseModel):
    prompt: str
    num_samples: int = 1
    max_new_tokens: int = 60
    top_k: int = 100
    # temperature: float = 0.8
    temperature: float = 1e-6
    # seed: Optional[int] = None
    seed: Optional[int] = 1234


class Token(BaseModel):
    text: str
    logprob: float
    top_logprob: Dict[str, float]


class ProcessResponse(BaseModel):
    text: str
    tokens: List[Token]
    logprob: float
    request_time: float


class TokenizeRequest(BaseModel):
    text: str
    truncation: bool = True
    max_length: int = 2048


class TokenizeResponse(BaseModel):
    tokens: List[int]
    request_time: float
