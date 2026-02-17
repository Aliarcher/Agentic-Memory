from typing import Optional
from langchain_openai import ChatOpenAI
import logging

from config.settings import settings
from core.models.state import AgentState
from memory import WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from providers.weaviate import WeaviateProvider

class MemoryAgent:
    """Main agent orchestrating all memory systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state = AgentState()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=settings.TEMPERATURE,
            model=settings.MODEL_NAME
        )
        
        # Initialize memory systems
        self.working_memory = WorkingMemory()
        self.provider = WeaviateProvider()
        self.episodic_memory = EpisodicMemory(self.provider, self.llm)
        self.semantic_memory = SemanticMemory(self.provider)
        self.procedural_memory = ProceduralMemory(self.llm)
        
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize agent and all components"""
        if not self.initialized:
            await self.provider.initialize()
            await self.procedural_memory.initialize()
            self.initialized = True
            self.logger.info("Agent initialized successfully")
    
    async def process_message(self, user_input: str) -> str:
        """Process a user message through all memory systems"""
        if not self.initialized:
            await self.initialize()
        
        # Retrieve relevant memories
        episodic = await self.episodic_memory.retrieve(user_input)
        semantic = await self.semantic_memory.retrieve(user_input)
        procedural = await self.procedural_memory.retrieve()
        
        # Create and store system prompt
        system_prompt = await self._create_system_prompt(episodic, procedural)
        await self.working_memory.store_system(system_prompt)
        
        # Add semantic context and user message
        if semantic:
            await self.working_memory.store_semantic(semantic)
        await self.working_memory.store_user(user_input)
        
        # Generate response
        response = await self.llm.ainvoke(self.working_memory.get_messages())
        await self.working_memory.store_ai(response.content)
        
        # Update state
        self._update_state(episodic)
        
        return response.content
    
    async def end_conversation(self) -> None:
        """End conversation and update long-term memory"""
        # Get conversation history
        messages = self.working_memory.get_messages(exclude_system=True)
        
        # Store in episodic memory
        await self.episodic_memory.store(messages)
        
        # Update procedural memory
        await self.procedural_memory.update(
            list(self.state.what_worked),
            list(self.state.what_to_avoid)
        )
        
        # Clear working memory
        await self.working_memory.clear()
        self.state.reset()
        
        self.logger.info("Conversation ended, memories updated")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown agent"""
        await self.provider.close()
        self.logger.info("Agent shutdown complete")