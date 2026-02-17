from fastapi import Request
from agent.core import MemoryAgent
from agent.conversation import ConversationManager

async def get_agent(request: Request) -> MemoryAgent:
    """Dependency to get agent instance"""
    return request.app.state.agent

async def get_conversation_manager(request: Request) -> ConversationManager:
    """Dependency to get conversation manager"""
    agent = await get_agent(request)
    
    # Create new conversation manager for each request
    manager = ConversationManager(agent)
    await manager.start()
    
    return manager