# app/prompts.py
"""
Centralized Prompt Engineering for Memory Layer System

DESIGN PRINCIPLES:
1. **Clarity over cleverness** - Direct, unambiguous instructions
2. **Structure** - Clear sections, bullet points, examples
3. **Consistency** - Similar formatting across prompts
4. **Constraints** - Explicit rules to prevent hallucination
5. **Context-awareness** - Include relevant examples

PROMPT TYPES:
- Decision: Should we query KG?
- System: Main chat assistant behavior
- Summarization: Extract facts from conversations
- Extraction: Structured data extraction

OPTIMIZATION FOR:
- GPT-4 (primary)
- GPT-3.5-turbo (fallback)
- Temperature tuning per use case
"""

# =============================================================================
# DECISION PROMPTS - Determine when to query Knowledge Graph
# =============================================================================

DECISION_PROMPT_TEMPLATE = """Classify if this query needs the Knowledge Graph (user's long-term memory).

**Query KG (YES) if:**
- References past interactions, preferences, or history
- Contains memory keywords: remember, told you, last time, before, previous
- Asks about user-specific information: "my", "what did I", "do I like"
- Requires personalized context

**Skip KG (NO) if:**
- General knowledge questions
- Greetings or small talk
- Real-time information requests
- Definitions or explanations
- No reference to past

**Output:** Only 'YES' or 'NO'

Query: {user_input}"""

# Alternative: More conservative (fewer KG queries)
DECISION_PROMPT_CONSERVATIVE = """Classify if this query needs the Knowledge Graph.
Answer YES only if it explicitly references:
- Past conversations ("you said", "we talked about")
- Personal preferences ("my favorite", "I like")
- Specific facts about the user

Otherwise answer NO.

User: {user_input}"""

# =============================================================================
# SYSTEM PROMPTS - Main conversation assistant
# =============================================================================

SYSTEM_PROMPT_BASE = """You are an AI assistant with long-term memory capabilities.

**Key Principles:**
1. **Natural conversation** - Friendly, adaptive to user's style
2. **Accuracy first** - Don't make assumptions or invent facts
3. **Completeness** - Thorough responses when needed
4. **Language matching** - Respond in user's language

**Code Generation:**
- Complete, runnable implementations
- All imports and dependencies included
- Clear comments in user's language

**Memory Honesty:**
If asked about past interactions without available memory context, acknowledge the gap honestly.
Do not fabricate or infer information not explicitly stored."""

SYSTEM_PROMPT_WITH_MEMORIES = """You are an AI assistant with long-term memory capabilities.

**Key Principles:**
1. **Natural conversation** - Friendly, adaptive to user's style
2. **Accuracy first** - Don't make assumptions or invent facts
3. **Completeness** - Thorough responses when needed
4. **Language matching** - Respond in user's language

**Code Generation:**
- Complete, runnable implementations
- All imports and dependencies included
- Clear comments in user's language

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 RETRIEVED MEMORIES (Knowledge Graph)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**CRITICAL RULES:**
✓ ONLY use facts explicitly stated below
✓ Weave memories naturally into conversation
✓ If asked about something NOT below, acknowledge the gap
✗ Do NOT infer or extrapolate beyond stated facts
✗ Do NOT make assumptions from partial information

**Retrieved Facts:**
{facts_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For queries about past interactions NOT covered above, respond:
"I don't have that specific information in my current memory context.\""""

# =============================================================================
# SUMMARIZATION PROMPTS - Extract facts from conversations
# =============================================================================

SUMMARIZATION_PROMPT_TEMPLATE = """Extract ALL important facts from this conversation.
Each fact should be SPECIFIC, CONCRETE, and ACTIONABLE.

**Quality Criteria - Think Step by Step:**
1. Is this fact PERSISTENT? (Not temporary like greetings)
2. Is this fact SPECIFIC? (Not vague like "discussed something")
3. Does it include CONTEXT? (Who, what, why, where, when)
4. Is it ACTIONABLE? (Useful for future reference)

**Format:** One fact per line, starting with "-"
**Language:** Always write facts in ENGLISH for consistency

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**GOOD Examples (Specific & Contextual):**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Identity & Background:
- User's name is Nguyên Trịnh and prefers to be called Nguyên
- User is from Hà Nội, Vietnam and currently lives there
- User is 25 years old and works as a software developer at VinTech

Preferences & Interests:
- User likes to eat cơm tấm (broken rice) with mắm tôm sauce
- User prefers Python over C for rapid prototyping due to simpler syntax
- User enjoys playing strategy games like Civilization VI in free time

Technical Knowledge:
- User learned Python async programming for handling I/O-bound tasks
- User implemented A* pathfinding algorithm in snake game for AI opponent
- User solved font rendering errors by switching to Arial font family
- User understands Python GIL limits multithreading performance for CPU-bound tasks

Projects & Actions:
- User developed snake game with wrap-around screen edges feature
- User added obstacles feature to snake game to increase difficulty
- User requested code review for Django REST API authentication module

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**BAD Examples (Avoid These):**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ "User said hello" 
   → Not persistent, just greeting

❌ "User discussed programming"
   → Too vague - what about programming?
   ✓ Better: "User learned Python async programming for I/O-bound tasks"

❌ "User prefers Python"
   → Missing context - why? for what?
   ✓ Better: "User prefers Python over C for web development due to faster iteration"

❌ "User enhanced the game"
   → Too vague - what enhancements?
   ✓ Better: "User added obstacles and AI pathfinding to snake game"

❌ "User is interested in AI"
   → Vague interest
   ✓ Better: "User is learning machine learning with TensorFlow for image classification projects"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Instructions:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Extract 1-5 facts (as many as are truly important)
2. Be CONCRETE: Include specific names, technologies, reasons
3. Include CONTEXT: Add "why", "for what purpose", "in what way"
4. Preserve TECHNICAL TERMS: Keep specific libraries, tools, concepts
5. Skip: greetings, small talk, meta-conversation, vague statements
6. Write in ENGLISH regardless of conversation language

**Think before extracting:**
- Would this fact be useful in a future conversation?
- Is it specific enough to be actionable?
- Does it capture the "why" and "how", not just "what"?

**Conversation:**
{conversation}

**Extracted Facts in English (specific, concrete, contextual):**"""

# Alternative: Detailed multi-fact extraction
SUMMARIZATION_PROMPT_DETAILED = """Extract key facts from this conversation.

**Focus Areas:**
1. Identity (name, role, location)
2. Preferences & interests (with reasons)
3. Relationships & connections
4. Problems & solutions
5. Plans & goals

**Format:** Concise paragraph or bullet points

**Conversation:**
{conversation}

**Summary:**"""

# =============================================================================
# FACT EXTRACTION PROMPTS - For structured data extraction
# =============================================================================

FACT_EXTRACTION_PROMPT = """Extract structured knowledge triplets.

**Format:** Subject | Relationship | Object

**Examples:**
- User | solved [problem] using | [solution]
- User | prefers | [X] over [Y]
- User | works as | [role]
- [Person A] | knows | [Person B]

**Conversation:**
{conversation}

**Triplets:**"""

# =============================================================================
# QUERY EXPANSION PROMPTS - Improve search queries
# =============================================================================

QUERY_EXPANSION_PROMPT = """Generate 3 semantic variations of this query for better search recall.

**Techniques:**
- Synonyms & related terms
- Different phrasings
- Broader/narrower concepts
- Language variations

**Original:** {query}

**Variations:**
1.
2.
3."""

# =============================================================================
# TRANSLATION PROMPTS - For multilingual search
# =============================================================================

QUERY_TRANSLATION_PROMPT = """Translate this query to English for knowledge graph search.

**Rules:**
- Preserve semantic meaning
- Keep technical terms
- Maintain query intent
- Return only the translation

**Original Query:** {query}

**English Translation:**"""

# =============================================================================
# ENTITY EXTRACTION PROMPTS - Extract named entities
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """Extract named entities with categories.

**Categories:**
- PERSON (names, roles)
- LOCATION (places, addresses)
- ORGANIZATION (companies, groups)
- CONCEPT (ideas, methods, technologies)
- OTHER (if significant)

**Format:** entity | category

**Text:**
{text}

**Entities:**"""

# =============================================================================
# PROMPT UTILITIES
# =============================================================================

def format_decision_prompt(user_input: str, conservative: bool = False) -> str:
    """Format the decision prompt with user input."""
    template = DECISION_PROMPT_CONSERVATIVE if conservative else DECISION_PROMPT_TEMPLATE
    return template.format(user_input=user_input)

def format_system_prompt(facts: list[str] = None) -> str:
    """Format the system prompt with optional facts from KG."""
    if not facts:
        return SYSTEM_PROMPT_BASE
    
    facts_context = "\n".join(f"- {fact}" for fact in facts)
    return SYSTEM_PROMPT_WITH_MEMORIES.format(facts_context=facts_context)

def format_summarization_prompt(conversation: list[str], detailed: bool = False) -> str:
    """Format the summarization prompt with conversation history."""
    conv_text = "\n".join(conversation)
    template = SUMMARIZATION_PROMPT_DETAILED if detailed else SUMMARIZATION_PROMPT_TEMPLATE
    return template.format(conversation=conv_text)

def format_fact_extraction_prompt(conversation: str) -> str:
    """Format the fact extraction prompt."""
    return FACT_EXTRACTION_PROMPT.format(conversation=conversation)

def format_query_expansion_prompt(query: str) -> str:
    """Format the query expansion prompt."""
    return QUERY_EXPANSION_PROMPT.format(query=query)

def format_entity_extraction_prompt(text: str) -> str:
    """Format the entity extraction prompt."""
    return ENTITY_EXTRACTION_PROMPT.format(text=text)

def format_query_translation_prompt(query: str) -> str:
    """Format the query translation prompt."""
    return QUERY_TRANSLATION_PROMPT.format(query=query)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROMPT_CONFIG = {
    "decision": {
        "temperature": 0.0,
        "max_tokens": 10,
        "conservative_mode": False,
    },
    "chat": {
        "temperature": 0.7,  # Higher for more creative responses
        "max_tokens": 5000,  
    },
    "summarization": {
        "temperature": 0.2,  # Lower for more factual extraction
        "max_tokens": 250,   # Increased to support multiple facts
        "detailed_mode": False,
    },
    "fact_extraction": {
        "temperature": 0.1,
        "max_tokens": 200,
    },
    "translation": {
        "temperature": 0.2,  # Low for accurate translation
        "max_tokens": 100,
    }
}

def get_prompt_config(prompt_type: str) -> dict:
    """Get configuration for a specific prompt type."""
    return PROMPT_CONFIG.get(prompt_type, {})
