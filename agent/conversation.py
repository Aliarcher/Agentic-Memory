from typing import Optional, Dict, Any
import logging
from datetime import datetime
from .core import MemoryAgent
from core.exceptions import AgentError
from core.models.state import AgentState

class ConversationManager:
    """Manages conversation flow and lifecycle"""
    
    def __init__(self, agent: MemoryAgent):
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            "messages": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def start(self) -> None:
        """Start a new conversation"""
        self.stats["start_time"] = datetime.now()
        self.logger.info(f"Starting conversation {self.conversation_id}")
        
        if not self.agent.initialized:
            await self.agent.initialize()
    
    async def process(self, user_input: str) -> str:
        """Process a single message"""
        try:
            self.stats["messages"] += 1
            response = await self.agent.process_message(user_input)
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            raise AgentError(f"Failed to process message: {e}")
    
    async def end(self) -> Dict[str, Any]:
        """End conversation and return stats"""
        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        await self.agent.end_conversation()
        
        summary = {
            "conversation_id": self.conversation_id,
            "duration_seconds": duration,
            "total_messages": self.stats["messages"],
            "avg_response_time": duration / max(self.stats["messages"], 1)
        }
        
        self.logger.info(f"Ended conversation {self.conversation_id}: {summary}")
        return summary
    
    async def reset(self) -> None:
        """Reset conversation state"""
        await self.agent.end_conversation()
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {"messages": 0, "start_time": None, "end_time": None}
        self.logger.info(f"Reset conversation, new ID: {self.conversation_id}")