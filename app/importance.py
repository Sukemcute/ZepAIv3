"""
Importance scoring for memory facts
Score facts by their relevance and usefulness (0.0 - 1.0)
"""

import os
from openai import OpenAI
from typing import Dict

# Importance categories with base scores
IMPORTANCE_CATEGORIES = {
    "identity": 1.0,       # Name, age, location, personal info
    "preference": 0.8,     # Likes, dislikes, interests  
    "knowledge": 0.7,      # Learned topics, understanding
    "action": 0.6,         # Things done, completed tasks
    "opinion": 0.5,        # Views, thoughts on topics
    "question": 0.3,       # Questions asked
    "greeting": 0.1,       # Small talk, greetings
}

IMPORTANCE_SCORING_PROMPT = """Classify this fact's importance category and score.

**Fact:** "{fact}"

**Categories (score):**
- identity (1.0): Personal information like name, age, location, occupation
- preference (0.8): Likes, dislikes, interests, favorites
- knowledge (0.7): Skills learned, topics understood, technical knowledge
- action (0.6): Concrete things done, projects completed
- opinion (0.5): Views, thoughts, beliefs on topics
- question (0.3): Questions asked (usually not persistent)
- greeting (0.1): Greetings, small talk, meta-conversation

**Examples:**

Fact: "User's name is John Smith"
→ identity|1.0

Fact: "User prefers Python over Java for web development"
→ preference|0.8

Fact: "User learned async programming in Python"
→ knowledge|0.7

Fact: "User completed snake game project with AI opponent"
→ action|0.6

Fact: "User said hello"
→ greeting|0.1

**Output format:** category|score

**Your classification:**"""


class ImportanceScorer:
    """Score facts by importance for retrieval prioritization"""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    async def score_fact(self, fact: str) -> Dict[str, float]:
        """
        Score a fact's importance
        
        Returns:
            {
                "category": "identity|preference|knowledge|...",
                "score": 0.0-1.0,
                "reasoning": "why this category"
            }
        """
        try:
            prompt = IMPORTANCE_SCORING_PROMPT.format(fact=fact)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low for consistent scoring
                max_tokens=50,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse "category|score"
            if "|" in result:
                category, score_str = result.split("|", 1)
                category = category.strip().lower()
                score = float(score_str.strip())
                
                return {
                    "category": category,
                    "score": score,
                    "reasoning": f"Classified as {category}"
                }
            else:
                # Fallback to default
                return {
                    "category": "knowledge",
                    "score": 0.7,
                    "reasoning": "Default score (parsing failed)"
                }
                
        except Exception as e:
            # Fallback to heuristic scoring
            return self._heuristic_score(fact)
    
    def _heuristic_score(self, fact: str) -> Dict[str, float]:
        """Fast heuristic scoring without LLM call"""
        fact_lower = fact.lower()
        
        # Identity keywords
        if any(kw in fact_lower for kw in ["name is", "called", "age", "from", "live in", "work at"]):
            return {"category": "identity", "score": 1.0, "reasoning": "Contains identity keywords"}
        
        # Preference keywords
        if any(kw in fact_lower for kw in ["like", "prefer", "favorite", "enjoy", "hate", "dislike"]):
            return {"category": "preference", "score": 0.8, "reasoning": "Contains preference keywords"}
        
        # Knowledge keywords
        if any(kw in fact_lower for kw in ["learned", "understand", "know", "studied", "mastered"]):
            return {"category": "knowledge", "score": 0.7, "reasoning": "Contains learning keywords"}
        
        # Action keywords
        if any(kw in fact_lower for kw in ["completed", "built", "created", "developed", "implemented"]):
            return {"category": "action", "score": 0.6, "reasoning": "Contains action keywords"}
        
        # Greeting keywords
        if any(kw in fact_lower for kw in ["hello", "hi", "said goodbye", "greeted"]):
            return {"category": "greeting", "score": 0.1, "reasoning": "Greeting/small talk"}
        
        # Default: opinion/generic
        return {"category": "opinion", "score": 0.5, "reasoning": "Default heuristic score"}
    
    def should_ingest(self, fact: str, threshold: float = 0.3) -> tuple[bool, Dict]:
        """
        Determine if fact should be ingested based on importance
        
        Args:
            fact: The fact to evaluate
            threshold: Minimum importance score (default 0.3)
            
        Returns:
            (should_ingest: bool, score_info: dict)
        """
        score_info = self._heuristic_score(fact)
        should_ingest = score_info["score"] >= threshold
        
        return should_ingest, score_info


# Singleton instance
_scorer = None

def get_scorer() -> ImportanceScorer:
    """Get global importance scorer instance"""
    global _scorer
    if _scorer is None:
        _scorer = ImportanceScorer()
    return _scorer
