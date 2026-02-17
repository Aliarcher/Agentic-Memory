from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .dependencies import get_agent, get_conversation_manager
from agent.core import MemoryAgent
from agent.conversation import ConversationManager

router = APIRouter()

class MessageRequest(BaseModel):
    """Message request model"""
    message: str
    conversation_id: Optional[str] = None

class MessageResponse(BaseModel):
    """Message response model"""
    response: str
    conversation_id: str
    metadata: Dict[str, Any]

class MemoryQuery(BaseModel):
    """Memory query model"""
    query: str
    memory_type: str  # episodic, semantic, procedural
    limit: int = 5

class MemoryResponse(BaseModel):
    """Memory response model"""
    results: List[Dict[str, Any]]
    count: int

@router.post("/chat", response_model=MessageResponse)
async def chat(
    request: MessageRequest,
    background_tasks: BackgroundTasks,
    agent: MemoryAgent = Depends(get_agent),
    conversation: ConversationManager = Depends(get_conversation_manager)
):
    """Send a message to the agent"""
    try:
        response = await agent.process_message(request.message)
        
        return MessageResponse(
            response=response,
            conversation_id=conversation.conversation_id,
            metadata={
                "message_count": conversation.stats["messages"],
                "timestamp": str(conversation.stats.get("start_time"))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conversation/end")
async def end_conversation(
    background_tasks: BackgroundTasks,
    conversation: ConversationManager = Depends(get_conversation_manager)
):
    """End current conversation"""
    try:
        summary = await conversation.end()
        return {"status": "ended", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory/{memory_type}")
async def search_memory(
    memory_type: str,
    query: str,
    limit: int = 5,
    agent: MemoryAgent = Depends(get_agent)
):
    """Search specific memory type"""
    try:
        if memory_type == "episodic":
            result = await agent.episodic_memory.retrieve(query, limit=limit)
        elif memory_type == "semantic":
            result = await agent.semantic_memory.retrieve(query, limit=limit)
        elif memory_type == "procedural":
            result = await agent.procedural_memory.retrieve()
        else:
            raise HTTPException(status_code=400, detail="Invalid memory type")
        
        return MemoryResponse(
            results=[result] if result else [],
            count=1 if result else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/memory/{memory_type}")
async def clear_memory(
    memory_type: str,
    agent: MemoryAgent = Depends(get_agent)
):
    """Clear specific memory type"""
    try:
        if memory_type == "working":
            await agent.working_memory.clear()
        elif memory_type == "episodic":
            # Clear episodic memory collection
            pass
        else:
            raise HTTPException(status_code=400, detail="Invalid memory type")
        
        return {"status": "cleared", "memory_type": memory_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats(
    agent: MemoryAgent = Depends(get_agent)
):
    """Get agent statistics"""
    return {
        "initialized": agent.initialized,
        "working_memory_size": len(agent.working_memory._messages),
        "state": agent.state.to_dict()
    }