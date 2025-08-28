# Data classes for storing vector embeddings
from pydantic import BaseModel
from typing import List

class EmbeddingRecord(BaseModel):
    item_id: str
    text: str
    embedding: List[float]
    metadata: dict
