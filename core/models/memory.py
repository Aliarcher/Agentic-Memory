from pydantic import BaseModel, Field
from typing import List, Optional, Set,Dict
from datetime import datetime

class EpisodicMemoryEntry(BaseModel):
    """Model for episodic memory entries"""
    id: Optional[str] = None
    conversation: str
    context_tags: List[str] = Field(default_factory=list)
    conversation_summary: str = ""
    what_worked: str = ""
    what_to_avoid: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0

class ReflectionResult(BaseModel):
    """Result of memory reflection"""
    context_tags: List[str]
    conversation_summary: str
    what_worked: str
    what_to_avoid: str

class SemanticChunk(BaseModel):
    """Semantic memory chunk"""
    id: str
    content: str
    source: str
    chunk_index: int
    metadata: Dict = Field(default_factory=dict)

class ProceduralRule(BaseModel):
    """Procedural memory rule"""
    index: int
    instruction: str
    rationale: str
    category: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)