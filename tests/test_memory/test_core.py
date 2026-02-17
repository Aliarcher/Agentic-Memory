import pytest
from unittest.mock import AsyncMock, Mock, patch
from agent.core import MemoryAgent
from core.models.state import AgentState

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization"""
    assert agent.initialized is True
    assert agent.state is not None
    assert agent.llm is not None

@pytest.mark.asyncio
async def test_agent_process_message(agent, mock_provider):
    """Test processing a message"""
    # Mock memory retrievals
    agent.episodic_memory.retrieve = AsyncMock(return_value=None)
    agent.semantic_memory.retrieve = AsyncMock(return_value="Semantic context")
    agent.procedural_memory.retrieve = AsyncMock(return_value="Procedural rules")
    
    # Mock working memory methods
    agent.working_memory.store_system = AsyncMock()
    agent.working_memory.store_semantic = AsyncMock()
    agent.working_memory.store_user = AsyncMock()
    agent.working_memory.store_ai = AsyncMock()
    agent.working_memory.get_messages = AsyncMock(return_value=[])
    
    response = await agent.process_message("Hello")
    
    assert response == "Test response"
    agent.working_memory.store_system.assert_called_once()
    agent.working_memory.store_user.assert_called_once_with("Hello")
    agent.working_memory.store_ai.assert_called_once()

@pytest.mark.asyncio
async def test_agent_end_conversation(agent):
    """Test ending conversation"""
    agent.working_memory.get_messages = AsyncMock(return_value=[])
    agent.working_memory.clear = AsyncMock()
    agent.episodic_memory.store = AsyncMock()
    agent.procedural_memory.update = AsyncMock()
    
    await agent.end_conversation()
    
    agent.episodic_memory.store.assert_called_once()
    agent.procedural_memory.update.assert_called_once()
    agent.working_memory.clear.assert_called_once()

@pytest.mark.asyncio
async def test_agent_state_management(agent):
    """Test state management"""
    # Initial state
    assert isinstance(agent.state, AgentState)
    assert agent.state.total_messages == 0
    
    # Update state with episodic memory
    class MockEpisodic:
        conversation_summary = "Test summary"
        what_worked = "Worked"
        what_to_avoid = "Avoid"
    
    agent._update_state(MockEpisodic())
    
    assert "Test summary" in agent.state.episodic_history
    assert "Worked" in agent.state.what_worked
    assert "Avoid" in agent.state.what_to_avoid

@pytest.mark.asyncio
async def test_agent_shutdown(agent, mock_provider):
    """Test agent shutdown"""
    agent.provider.close = AsyncMock()
    
    await agent.shutdown()
    
    agent.provider.close.assert_called_once()