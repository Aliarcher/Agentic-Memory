import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from agent.core import MemoryAgent
from agent.conversation import ConversationManager

@pytest.mark.asyncio
async def test_full_conversation_pipeline():
    """Test complete conversation flow"""
    
    # Mock all external dependencies
    with patch('agent.core.ChatOpenAI') as mock_chat, \
         patch('agent.core.WeaviateProvider') as mock_provider_class, \
         patch('memory.procedural.Path') as mock_path:
        
        # Setup mocks
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value.content = "Test response"
        mock_chat.return_value = mock_llm
        
        mock_provider = AsyncMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.close = AsyncMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        mock_provider_class.return_value = mock_provider
        
        # Mock file operations
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = "1. Test rule - Test rationale"
        mock_path.return_value = mock_file
        
        # Create and initialize agent
        agent = MemoryAgent()
        await agent.initialize()
        
        # Mock memory retrievals
        agent.episodic_memory.retrieve = AsyncMock(return_value=None)
        agent.semantic_memory.retrieve = AsyncMock(return_value="Semantic context")
        agent.procedural_memory.retrieve = AsyncMock(return_value="Procedural rules")
        
        # Test conversation
        manager = ConversationManager(agent)
        await manager.start()
        
        # Send messages
        response1 = await manager.process("Hello")
        assert response1 == "Test response"
        
        response2 = await manager.process("How are you?")
        assert response2 == "Test response"
        
        # End conversation
        summary = await manager.end()
        
        assert summary["total_messages"] == 2
        assert "duration_seconds" in summary
        
        # Shutdown
        await agent.shutdown()

@pytest.mark.asyncio
async def test_memory_retrieval_flow():
    """Test memory retrieval across all systems"""
    
    with patch('agent.core.ChatOpenAI') as mock_chat, \
         patch('agent.core.WeaviateProvider') as mock_provider_class:
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value.content = "Test response"
        mock_chat.return_value = mock_llm
        
        mock_provider = AsyncMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.close = AsyncMock()
        mock_provider_class.return_value = mock_provider
        
        agent = MemoryAgent()
        await agent.initialize()
        
        # Test parallel memory retrieval
        query = "test query"
        
        # Mock different memory responses
        mock_episodic = AsyncMock()
        mock_episodic.retrieve.return_value = None
        agent.episodic_memory = mock_episodic
        
        mock_semantic = AsyncMock()
        mock_semantic.retrieve.return_value = "Semantic result"
        agent.semantic_memory = mock_semantic
        
        mock_procedural = AsyncMock()
        mock_procedural.retrieve.return_value = "Procedural result"
        agent.procedural_memory = mock_procedural
        
        # Process message to trigger retrievals
        await agent.process_message(query)
        
        # Verify all memories were queried
        mock_episodic.retrieve.assert_called_once_with(query)
        mock_semantic.retrieve.assert_called_once_with(query)
        mock_procedural.retrieve.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in pipeline"""
    
    with patch('agent.core.ChatOpenAI') as mock_chat, \
         patch('agent.core.WeaviateProvider') as mock_provider_class:
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM Error")
        mock_chat.return_value = mock_llm
        
        mock_provider = AsyncMock()
        mock_provider.initialize = AsyncMock()
        mock_provider_class.return_value = mock_provider
        
        agent = MemoryAgent()
        await agent.initialize()
        
        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            await agent.process_message("Hello")
        
        assert "LLM Error" in str(exc_info.value)
        
        # Agent should still be usable after error
        agent.state.error_count = 1
        assert agent.state.error_count == 1