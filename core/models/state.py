from pydantic import BaseModel, Field
from typing import List, Set, Optional, Dict, Any
from datetime import datetime

class AgentState(BaseModel):
    """Current state of the agent"""
    
    # Session info
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    start_time: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    
    # Memory tracking
    episodic_history: List[str] = Field(default_factory=list)
    what_worked: Set[str] = Field(default_factory=set)
    what_to_avoid: Set[str] = Field(default_factory=set)
    current_context_tags: List[str] = Field(default_factory=list)
    
    # Performance metrics
    total_messages: int = 0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    
    # Flags
    is_active: bool = True
    has_error: bool = False
    error_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def add_episodic(self, summary: str):
        """Add to episodic history"""
        self.episodic_history.append(summary)
        if len(self.episodic_history) > 10:  # Keep last 10
            self.episodic_history = self.episodic_history[-10:]
    
    def add_what_worked(self, item: str):
        """Add what worked item"""
        self.what_worked.add(item)
    
    def add_what_to_avoid(self, item: str):
        """Add what to avoid item"""
        self.what_to_avoid.add(item)
    
    def reset(self):
        """Reset state for new conversation"""
        self.episodic_history = []
        self.what_worked = set()
        self.what_to_avoid = set()
        self.current_context_tags = []
        self.total_messages = 0
        self.has_error = False
        self.error_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "session_id": self.session_id,
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "avg_response_time": self.avg_response_time,
            "is_active": self.is_active,
            "has_error": self.has_error
        }