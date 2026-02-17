from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from core.interfaces.provider import MemoryProvider
import logging

class BaseMemoryProvider(MemoryProvider, ABC):
    """Base implementation of memory provider with common functionality"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
    
    @abstractmethod
    async def _connect(self) -> None:
        """Internal connection logic"""
        pass
    
    @abstractmethod
    async def _disconnect(self) -> None:
        """Internal disconnection logic"""
        pass
    
    async def initialize(self) -> None:
        """Initialize with connection retry logic"""
        if not self._initialized:
            try:
                await self._connect()
                self._initialized = True
                self.logger.info(f"{self.__class__.__name__} initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
                raise
    
    async def close(self) -> None:
        """Close with cleanup"""
        if self._initialized:
            try:
                await self._disconnect()
                self._initialized = False
                self.logger.info(f"{self.__class__.__name__} closed")
            except Exception as e:
                self.logger.error(f"Error closing {self.__class__.__name__}: {e}")
                raise
    
    async def health_check(self) -> bool:
        """Default health check"""
        return self._initialized
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._initialized:
            import asyncio
            try:
                asyncio.create_task(self.close())
            except:
                pass