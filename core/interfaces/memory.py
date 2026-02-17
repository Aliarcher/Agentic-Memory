from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from langchain_core.messages import BaseMessage

class MemoryInterface(ABC):
    """Base interface for all memory types"""
    
    @abstractmethod
    async def store(self, data: Any, **kwargs) -> None:
        """Store data in memory"""
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, **kwargs) -> Any:
        """Retrieve data from memory"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear memory contents"""
        pass

class WorkingMemoryInterface(MemoryInterface):
    @abstractmethod
    async def get_context(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """Get current working memory context"""
        pass

class EpisodicMemoryInterface(MemoryInterface):
    @abstractmethod
    async def reflect(self, conversation: List[BaseMessage]) -> Dict:
        """Generate reflection from conversation"""
        pass

class SemanticMemoryInterface(MemoryInterface):
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> List[str]:
        """Search semantic memory"""
        pass

class ProceduralMemoryInterface(MemoryInterface):
    @abstractmethod
    async def update(self, new_rules: List[str]) -> None:
        """Update procedural rules"""
        pass