from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import logging
from datetime import datetime

from core.interfaces.memory import EpisodicMemoryInterface
from core.models.memory import EpisodicMemoryEntry, ReflectionResult
from providers.weaviate import WeaviateProvider
from core.exceptions import EpisodicMemoryError
from config.settings import settings

class EpisodicMemory(EpisodicMemoryInterface):
    """Episodic memory implementation for storing conversation experiences"""
    
    def __init__(self, provider: WeaviateProvider, llm: ChatOpenAI):
        self.provider = provider
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.collection_name = settings.EPISODIC_COLLECTION
        self.reflection_chain = self._create_reflection_chain()
    
    def _create_reflection_chain(self):
        """Create the reflection prompt chain"""
        template = """You are analyzing conversations about research papers to create memories that will help guide future interactions. Your task is to extract key elements that would be most helpful when encountering similar academic discussions in the future.

Review the conversation and create a memory reflection following these rules:
1. For any field where you don't have enough information or the field isn't relevant, use "N/A"
2. Be extremely concise - each string should be one clear, actionable sentence
3. Focus only on information that would be useful for handling similar future conversations
4. Context_tags should be specific enough to match similar situations but general enough to be reusable

Output valid JSON in exactly this format:
{{
    "context_tags": [string, string, ...],
    "conversation_summary": string,
    "what_worked": string,
    "what_to_avoid": string
}}

Here is the prior conversation:
{conversation}
"""
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | JsonOutputParser()
    
    async def store(self, messages: List[BaseMessage], **kwargs) -> None:
        """Store conversation as episodic memory"""
        try:
            # Format conversation
            conversation = self._format_conversation(messages)
            
            # Create reflection
            reflection = await self.reflection_chain.ainvoke(
                {"conversation": conversation}
            )
            
            # Create entry
            entry = EpisodicMemoryEntry(
                conversation=conversation,
                context_tags=reflection.get("context_tags", []),
                conversation_summary=reflection.get("conversation_summary", ""),
                what_worked=reflection.get("what_worked", ""),
                what_to_avoid=reflection.get("what_to_avoid", "")
            )
            
            # Store in database
            collection = self.provider.get_collection(self.collection_name)
            
            result = collection.data.insert({
                "conversation": entry.conversation,
                "context_tags": entry.context_tags,
                "conversation_summary": entry.conversation_summary,
                "what_worked": entry.what_worked,
                "what_to_avoid": entry.what_to_avoid,
                "created_at": entry.created_at.isoformat()
            })
            
            self.logger.info(f"Stored episodic memory with ID: {result}")
            
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to store episodic memory: {e}")
    
    async def retrieve(self, query: str, **kwargs) -> Optional[EpisodicMemoryEntry]:
        """Retrieve relevant episodic memories"""
        try:
            limit = kwargs.get("limit", 1)
            collection = self.provider.get_collection(self.collection_name)
            
            result = collection.query.hybrid(
                query=query,
                alpha=0.5,
                limit=limit
            )
            
            if result.objects:
                # Get the most relevant
                obj = result.objects[0]
                props = obj.properties
                
                # Update access stats
                collection.data.update(
                    uuid=obj.uuid,
                    properties={
                        "last_accessed": datetime.now().isoformat(),
                        "access_count": props.get("access_count", 0) + 1
                    }
                )
                
                return EpisodicMemoryEntry(
                    id=str(obj.uuid),
                    conversation=props.get("conversation", ""),
                    context_tags=props.get("context_tags", []),
                    conversation_summary=props.get("conversation_summary", ""),
                    what_worked=props.get("what_worked", ""),
                    what_to_avoid=props.get("what_to_avoid", ""),
                    created_at=datetime.fromisoformat(props.get("created_at", datetime.now().isoformat())),
                    last_accessed=datetime.fromisoformat(props.get("last_accessed", datetime.now().isoformat())),
                    access_count=props.get("access_count", 0)
                )
            
            return None
            
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to retrieve episodic memory: {e}")
    
    async def search_by_tags(self, tags: List[str], limit: int = 5) -> List[EpisodicMemoryEntry]:
        """Search memories by context tags"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            
            # Build filter for tags
            filter_conditions = []
            for tag in tags:
                filter_conditions.append({
                    "path": ["context_tags"],
                    "operator": "ContainsAny",
                    "valueText": tag
                })
            
            result = collection.query.fetch_objects(
                where={
                    "operator": "Or",
                    "operands": filter_conditions
                } if filter_conditions else None,
                limit=limit
            )
            
            entries = []
            for obj in result.objects:
                props = obj.properties
                entries.append(EpisodicMemoryEntry(
                    id=str(obj.uuid),
                    conversation=props.get("conversation", ""),
                    context_tags=props.get("context_tags", []),
                    conversation_summary=props.get("conversation_summary", ""),
                    what_worked=props.get("what_worked", ""),
                    what_to_avoid=props.get("what_to_avoid", "")
                ))
            
            return entries
            
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to search by tags: {e}")
    
    async def reflect(self, conversation: List[BaseMessage]) -> Dict:
        """Generate reflection from conversation"""
        try:
            formatted = self._format_conversation(conversation)
            return await self.reflection_chain.ainvoke({"conversation": formatted})
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to reflect on conversation: {e}")
    
    async def clear(self) -> None:
        """Clear all episodic memories"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            collection.data.delete_many({})
            self.logger.info("Cleared all episodic memories")
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to clear episodic memory: {e}")
    
    async def delete(self, memory_id: str) -> None:
        """Delete specific memory by ID"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            collection.data.delete_by_id(memory_id)
            self.logger.info(f"Deleted episodic memory: {memory_id}")
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to delete memory {memory_id}: {e}")
    
    def _format_conversation(self, messages: List[BaseMessage]) -> str:
        """Format messages for storage"""
        lines = []
        for msg in messages:
            if not isinstance(msg, SystemMessage):
                role = "HUMAN" if msg.type == "human" else "AI"
                lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            count = collection.aggregate.over_all(total_count=True)
            
            return {
                "total_memories": count.total_count,
                "collection": self.collection_name
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}