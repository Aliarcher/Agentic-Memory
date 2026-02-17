from typing import List, Optional, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from core.interfaces.memory import WorkingMemoryInterface
from core.exceptions import WorkingMemoryError
import logging

class WorkingMemory(WorkingMemoryInterface):
    """Working memory implementation for active conversation context"""
    
    def __init__(self, max_size: Optional[int] = 50, config: Optional[Dict] = None):
        self._messages: List[BaseMessage] = []
        self.max_size = max_size
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self._metadata: Dict[str, Any] = {
            "total_messages": 0,
            "system_prompts": 0,
            "user_messages": 0,
            "ai_messages": 0
        }
    
    async def store(self, message: BaseMessage, **kwargs) -> None:
        """Store a message in working memory"""
        try:
            self._messages.append(message)
            self._update_metadata(message)
            
            # Trim if exceeds max size
            if self.max_size and len(self._messages) > self.max_size:
                self._messages = self._messages[-self.max_size:]
                self.logger.debug(f"Trimmed working memory to {self.max_size} messages")
            
            self.logger.debug(f"Stored {message.type} message in working memory")
            
        except Exception as e:
            raise WorkingMemoryError(f"Failed to store message: {e}")
    
    async def store_user(self, content: str) -> None:
        """Store user message"""
        await self.store(HumanMessage(content=content))
    
    async def store_ai(self, content: str) -> None:
        """Store AI message"""
        await self.store(AIMessage(content=content))
    
    async def store_system(self, content: str) -> None:
        """Store system message"""
        await self.store(SystemMessage(content=content))
    
    async def store_semantic(self, content: str) -> None:
        """Store semantic context as human message"""
        await self.store(HumanMessage(content=f"[SEMANTIC CONTEXT]\n{content}"))
    
    async def retrieve(self, query: str = None, **kwargs) -> List[BaseMessage]:
        """Retrieve messages from working memory"""
        try:
            limit = kwargs.get('limit')
            msg_type = kwargs.get('type')
            
            messages = self._messages
            
            # Filter by type if specified
            if msg_type:
                if msg_type.lower() == 'system':
                    messages = [m for m in messages if isinstance(m, SystemMessage)]
                elif msg_type.lower() == 'user':
                    messages = [m for m in messages if isinstance(m, HumanMessage)]
                elif msg_type.lower() == 'ai':
                    messages = [m for m in messages if isinstance(m, AIMessage)]
            
            # Apply limit
            if limit:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            raise WorkingMemoryError(f"Failed to retrieve messages: {e}")
    
    async def get_context(self, limit: Optional[int] = None, exclude_system: bool = True) -> List[BaseMessage]:
        """Get working memory context"""
        if exclude_system:
            messages = [m for m in self._messages if not isinstance(m, SystemMessage)]
        else:
            messages = self._messages
        
        if limit:
            return messages[-limit:]
        return messages
    
    async def get_messages(self, exclude_system: bool = False) -> List[BaseMessage]:
        """Get all messages"""
        if exclude_system:
            return [m for m in self._messages if not isinstance(m, SystemMessage)]
        return self._messages.copy()
    
    async def clear(self) -> None:
        """Clear working memory"""
        self._messages.clear()
        self._metadata = {
            "total_messages": 0,
            "system_prompts": 0,
            "user_messages": 0,
            "ai_messages": 0
        }
        self.logger.debug("Working memory cleared")
    
    async def remove_last(self, n: int = 1) -> None:
        """Remove last n messages"""
        if n > 0 and n <= len(self._messages):
            removed = self._messages[-n:]
            self._messages = self._messages[:-n]
            
            # Update metadata
            for msg in removed:
                self._metadata["total_messages"] -= 1
                if isinstance(msg, SystemMessage):
                    self._metadata["system_prompts"] -= 1
                elif isinstance(msg, HumanMessage):
                    self._metadata["user_messages"] -= 1
                elif isinstance(msg, AIMessage):
                    self._metadata["ai_messages"] -= 1
            
            self.logger.debug(f"Removed last {n} messages")
    
    async def search(self, keyword: str) -> List[BaseMessage]:
        """Search messages containing keyword"""
        return [m for m in self._messages if keyword.lower() in m.content.lower()]
    
    async def get_last_user_message(self) -> Optional[HumanMessage]:
        """Get the last user message"""
        for msg in reversed(self._messages):
            if isinstance(msg, HumanMessage):
                return msg
        return None
    
    async def get_last_ai_message(self) -> Optional[AIMessage]:
        """Get the last AI message"""
        for msg in reversed(self._messages):
            if isinstance(msg, AIMessage):
                return msg
        return None
    
    def _update_metadata(self, message: BaseMessage) -> None:
        """Update metadata statistics"""
        self._metadata["total_messages"] += 1
        
        if isinstance(message, SystemMessage):
            self._metadata["system_prompts"] += 1
        elif isinstance(message, HumanMessage):
            self._metadata["user_messages"] += 1
        elif isinstance(message, AIMessage):
            self._metadata["ai_messages"] += 1
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get working memory metadata"""
        return {
            **self._metadata,
            "current_size": len(self._messages),
            "max_size": self.max_size,
            "utilization": len(self._messages) / self.max_size if self.max_size else 0
        }
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __getitem__(self, index: int) -> BaseMessage:
        return self._messages[index]