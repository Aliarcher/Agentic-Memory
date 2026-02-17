from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class MemoryProvider(ABC):
    """Base interface for memory storage providers"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close provider connection"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass
    
    @abstractmethod
    async def create_collection(self, name: str, schema: Dict[str, Any]) -> None:
        """Create a new collection"""
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection"""
        pass
    
    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all collections"""
        pass