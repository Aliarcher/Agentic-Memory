from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any
from datetime import datetime

def format_conversation(messages: List[BaseMessage], include_system: bool = False) -> str:
    """Format conversation for storage"""
    lines = []
    
    for msg in messages:
        if not include_system and isinstance(msg, SystemMessage):
            continue
        
        role = msg.type.upper()
        content = msg.content.strip()
        
        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."
        
        lines.append(f"{role}: {content}")
    
    return "\n".join(lines)

def format_memory_context(
    episodic: Dict[str, Any],
    semantic: str,
    procedural: str
) -> str:
    """Format memory context for prompt"""
    parts = []
    
    # Episodic context
    if episodic:
        parts.append("=== SIMILAR PAST CONVERSATIONS ===")
        if episodic.get("conversation_summary"):
            parts.append(f"Summary: {episodic['conversation_summary']}")
        if episodic.get("what_worked"):
            parts.append(f"What worked: {episodic['what_worked']}")
        if episodic.get("what_to_avoid"):
            parts.append(f"What to avoid: {episodic['what_to_avoid']}")
    
    # Semantic context
    if semantic:
        parts.append("\n=== RELEVANT KNOWLEDGE ===")
        parts.append(semantic)
    
    # Procedural context
    if procedural:
        parts.append("\n=== INTERACTION GUIDELINES ===")
        parts.append(procedural)
    
    return "\n".join(parts)

def format_procedural_rules(rules: List[str]) -> str:
    """Format procedural rules for display"""
    formatted = []
    for i, rule in enumerate(rules, 1):
        formatted.append(f"{i}. {rule}")
    return "\n".join(formatted)

def parse_procedural_rules(text: str) -> List[str]:
    """Parse procedural rules from text"""
    rules = []
    for line in text.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # Remove numbering/bullet
            clean_line = re.sub(r'^[\d\.\-\s]+', '', line)
            if clean_line:
                rules.append(clean_line)
    return rules

def format_response_metadata(
    response_time: float,
    tokens_used: int,
    memories_accessed: List[str]
) -> str:
    """Format response metadata"""
    meta = []
    meta.append(f"Response time: {response_time:.2f}s")
    meta.append(f"Tokens used: {tokens_used}")
    if memories_accessed:
        meta.append(f"Memories accessed: {', '.join(memories_accessed)}")
    return " | ".join(meta)

def to_json_serializable(obj: Any) -> Any:
    """Convert object to JSON serializable format"""
    if hasattr(obj, "dict"):
        return obj.dict()
    elif hasattr(obj, "__dict__"):
        return {k: to_json_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    return obj