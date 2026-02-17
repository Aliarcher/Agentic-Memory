from typing import List, Optional, Set, Dict, Any
from pathlib import Path
import logging
import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from core.interfaces.memory import ProceduralMemoryInterface
from core.models.memory import ProceduralRule
from core.exceptions import ProceduralMemoryError
from config.settings import settings

class ProceduralMemory(ProceduralMemoryInterface):
    """Procedural memory implementation for rules and guidelines"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.rules: List[ProceduralRule] = []
        self.file_path = settings.PROCEDURAL_MEMORY_PATH
        self.last_updated: Optional[datetime] = None
    
    async def initialize(self) -> None:
        """Load procedural rules from file"""
        try:
            if self.file_path.exists():
                content = self.file_path.read_text(encoding='utf-8')
                self.rules = self._parse_rules(content)
                self.logger.info(f"Loaded {len(self.rules)} procedural rules from {self.file_path}")
            else:
                # Create with default rules
                self.rules = self._get_default_rules()
                await self._save_rules()
                self.logger.info(f"Created default procedural rules at {self.file_path}")
            
            self.last_updated = datetime.now()
            
        except Exception as e:
            raise ProceduralMemoryError(f"Failed to initialize procedural memory: {e}")
    
    async def store(self, data: Any, **kwargs) -> None:
        """Store a new procedural rule"""
        try:
            rule = data if isinstance(data, ProceduralRule) else ProceduralRule(**data)
            self.rules.append(rule)
            await self._save_rules()
            self.logger.debug(f"Stored new procedural rule: {rule.instruction[:50]}...")
        except Exception as e:
            raise ProceduralMemoryError(f"Failed to store procedural rule: {e}")
    
    async def retrieve(self, query: str = None, **kwargs) -> str:
        """Get procedural rules as formatted string"""
        try:
            if not self.rules:
                return ""
            
            # Format rules for display
            formatted = []
            for i, rule in enumerate(self.rules, 1):
                formatted.append(f"{i}. {rule.instruction} - {rule.rationale}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            raise ProceduralMemoryError(f"Failed to retrieve procedural memory: {e}")
    
    async def update(self, what_worked: List[str], what_to_avoid: List[str]) -> None:
        """Update procedural rules based on new learnings"""
        try:
            # Create update prompt
            prompt = self._create_update_prompt(what_worked, what_to_avoid)
            
            # Generate new rules
            result = await self.llm.ainvoke(prompt)
            new_rules_text = result.content.strip()
            
            # Parse and update rules
            new_rules = self._parse_rules(new_rules_text)
            
            if new_rules:
                self.rules = new_rules[:10]  # Keep max 10 rules
                await self._save_rules()
                self.last_updated = datetime.now()
                self.logger.info(f"Updated procedural memory with {len(self.rules)} rules")
            
        except Exception as e:
            raise ProceduralMemoryError(f"Failed to update procedural memory: {e}")
    
    async def clear(self) -> None:
        """Clear all procedural rules"""
        self.rules = []
        await self._save_rules()
        self.logger.info("Cleared procedural memory")
    
    async def add_rule(self, instruction: str, rationale: str, category: Optional[str] = None) -> None:
        """Add a new rule"""
        rule = ProceduralRule(
            index=len(self.rules) + 1,
            instruction=instruction,
            rationale=rationale,
            category=category
        )
        self.rules.append(rule)
        await self._save_rules()
    
    async def remove_rule(self, index: int) -> None:
        """Remove a rule by index"""
        if 0 <= index - 1 < len(self.rules):
            removed = self.rules.pop(index - 1)
            # Re-index remaining rules
            for i, rule in enumerate(self.rules, 1):
                rule.index = i
            await self._save_rules()
            self.logger.info(f"Removed rule: {removed.instruction[:50]}...")
    
    async def search_rules(self, keyword: str) -> List[ProceduralRule]:
        """Search rules containing keyword"""
        keyword = keyword.lower()
        return [
            rule for rule in self.rules
            if keyword in rule.instruction.lower() or keyword in rule.rationale.lower()
        ]
    
    def _create_update_prompt(self, what_worked: List[str], what_to_avoid: List[str]) -> str:
        """Create prompt for updating rules"""
        current_rules = "\n".join([f"{i}. {r.instruction} - {r.rationale}" 
                                   for i, r in enumerate(self.rules, 1)])
        
        worked = "\n".join([f"- {item}" for item in what_worked if item and item != "N/A"])
        avoided = "\n".join([f"- {item}" for item in what_to_avoid if item and item != "N/A"])
        
        return f"""You are maintaining a continuously updated list of the most important procedural behavior instructions for an AI assistant. Your task is to refine and improve a list of key takeaways based on new conversation feedback while maintaining the most valuable existing insights.

CURRENT RULES:
{current_rules}

NEW FEEDBACK:
What Worked Well:
{worked}

What To Avoid:
{avoided}

Please generate an updated list of up to 10 key takeaways that combines:
1. The most valuable insights from the current takeaways
2. New learnings from the recent feedback
3. Any synthesized insights combining multiple learnings

Requirements for each takeaway:
- Must be specific and actionable
- Should address a distinct aspect of behavior
- Include a clear rationale
- Written in imperative form (e.g., "Maintain conversation context by...")

Format each takeaway as:
[#]. [Instruction] - [Brief rationale]

Return only the list, no preamble or explanation."""
    
    def _parse_rules(self, text: str) -> List[ProceduralRule]:
        """Parse rules from text"""
        rules = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Split instruction and rationale
            if ' - ' in line:
                instruction, rationale = line.split(' - ', 1)
            else:
                instruction = line
                rationale = ""
            
            rules.append(ProceduralRule(
                index=len(rules) + 1,
                instruction=instruction.strip(),
                rationale=rationale.strip()
            ))
        
        return rules
    
    def _get_default_rules(self) -> List[ProceduralRule]:
        """Get default procedural rules"""
        default_rules = [
            "Maintain conversation context by recalling previous interactions - Builds rapport and shows attention to user preferences over time.",
            "Use clear and concise language to convey information - Enhances understanding and avoids confusion.",
            "Offer structured breakdowns for complex topics - Facilitates comprehension and highlights key roles and functions.",
            "Ask clarifying questions when user requests are ambiguous - Ensures accurate assistance and reduces misunderstandings.",
            "Provide step-by-step guidance for complex tasks - Facilitates user comprehension and successful task completion.",
            "Acknowledge user emotions and respond empathetically - Builds trust and rapport with the user.",
            "Confirm and repeat the user's name to acknowledge recognition - Reinforces a personal connection and shows attentiveness.",
            "Offer alternative solutions when initial suggestions are not feasible - Demonstrates flexibility and commitment to user satisfaction.",
            "Provide specific suggestions tailored to the user's stated preferences - Shows attentiveness to user needs and enhances satisfaction.",
            "Continuously learn from user feedback to improve response quality - Enhances overall effectiveness and user experience."
        ]
        
        rules = []
        for i, rule_text in enumerate(default_rules, 1):
            if ' - ' in rule_text:
                instruction, rationale = rule_text.split(' - ', 1)
            else:
                instruction = rule_text
                rationale = ""
            
            rules.append(ProceduralRule(
                index=i,
                instruction=instruction.strip(),
                rationale=rationale.strip()
            ))
        
        return rules
    
    async def _save_rules(self) -> None:
        """Save rules to file"""
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format rules for saving
            lines = []
            for rule in self.rules:
                line = f"{rule.index}. {rule.instruction}"
                if rule.rationale:
                    line += f" - {rule.rationale}"
                lines.append(line)
            
            # Write to file
            self.file_path.write_text("\n".join(lines), encoding='utf-8')
            self.logger.debug(f"Saved {len(self.rules)} rules to {self.file_path}")
            
        except Exception as e:
            raise ProceduralMemoryError(f"Failed to save rules: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_rules": len(self.rules),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "file_path": str(self.file_path),
            "categories": list(set(r.category for r in self.rules if r.category))
        }