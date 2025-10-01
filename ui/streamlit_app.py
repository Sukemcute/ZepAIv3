import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List

import requests
import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path to import prompts module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.prompts import format_decision_prompt, format_system_prompt, format_summarization_prompt, PROMPT_CONFIG

# Load .env placed at memory_layer/.env so UI and API share config
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# OpenAI pricing (USD per 1M tokens) - updated Jan 2024
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost in USD based on token usage and model"""
    # Try to match model name (handle versioned names like gpt-4-0125-preview)
    for model_key, pricing in OPENAI_PRICING.items():
        if model_key in model:
            input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
            output_cost = (completion_tokens / 1_000_000) * pricing["output"]
            return input_cost + output_cost
    # Default fallback (use gpt-4o-mini pricing)
    return ((prompt_tokens / 1_000_000) * 0.150) + ((completion_tokens / 1_000_000) * 0.600)


def get_api_base_url() -> str:
    env_url = os.getenv("MEMORY_LAYER_API", "http://127.0.0.1:8000")
    return env_url.rstrip("/")


def post_json(path: str, payload: dict):
    base = get_api_base_url()
    url = f"{base}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return True, resp.json()
    except requests.RequestException as exc:
        return False, {"error": str(exc), "url": url}


def post_json_timeout(path: str, payload: dict, timeout_sec: int = 8):
    base = get_api_base_url()
    url = f"{base}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        return True, resp.json()
    except requests.RequestException as exc:
        return False, {"error": str(exc), "url": url}


def get_json(path: str):
    """GET request to API"""
    base = get_api_base_url()
    url = f"{base}{path}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return True, resp.json()
    except requests.RequestException as exc:
        return False, {"error": str(exc), "url": url}


st.set_page_config(page_title="Graphiti Memory Layer UI", page_icon="🧠", layout="centered")
st.title("🧠 Graphiti Memory Layer - Demo UI")
st.caption("Chat, ingest episodes and run search against the FastAPI server")

api_base = get_api_base_url()
st.info(f"API base: {api_base}")

tabs = st.tabs(["Chat", "Ingest", "Search", "Cache", "Debug"])

# ---------------------------- Chat Tab ----------------------------
with tabs[0]:
    st.subheader("Chat with Memory")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # list of {role, content, token_usage (optional)}

    # Initialize session states first
    if "group_id" not in st.session_state:
        import uuid
        st.session_state.group_id = f"chat-{uuid.uuid4()}"
    
    # Buffers for grouped ingest
    if "mem_buffer" not in st.session_state:
        st.session_state.mem_buffer = []  # list[str] lines "role: content"
    if "mem_user_count" not in st.session_state:
        st.session_state.mem_user_count = 0

    # Create chat container (displays at top, messages scroll)
    chat_container = st.container()
    
    # --- Settings and Controls (rendered first but display at bottom) ---
    st.markdown("---")  # Divider
    
    # Settings row
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        chat_name = st.text_input("Conversation name", value=st.session_state.get("chat_name", "agent_chat"), key="chat_name_input")
        st.session_state.chat_name = chat_name
    with col2:
        auto_save = st.checkbox("Auto-save turns", value=st.session_state.get("auto_save", True), key="auto_save_cb")
        st.session_state.auto_save = auto_save
    with col3:
        show_memories = st.checkbox("Show related memories", value=st.session_state.get("show_memories", True), key="show_mem_cb")
        st.session_state.show_memories = show_memories

    col4, col5, col6 = st.columns([1, 1, 1])
    with col4:
        pause_saving = st.checkbox("Pause saving to KG", value=st.session_state.get("pause_saving", False), key="pause_cb")
        st.session_state.pause_saving = pause_saving
    with col5:
        ingest_every_n_user_turns = st.selectbox("Mid-term: Ingest every N turns", [1, 2, 3, 5, 10], index=3, key="ingest_n_select",
                                                   help="N=1: Save each turn. N≥2: Summarize N turns into 1 fact")
        st.session_state.ingest_every_n = ingest_every_n_user_turns
    with col6:
        short_term_window = st.selectbox("Short-term: Keep last N turns", [3, 5, 10, 20], index=1, key="short_term_select",
                                          help="Number of recent conversation turns to keep as context")
        st.session_state.short_term_window = short_term_window
    
    col7, col8, col9 = st.columns([1, 1, 1])
    with col7:
        if st.button("Clear conversation", key="clear_btn"):
            st.session_state.chat_messages = []
            st.session_state.mem_buffer = []
            st.session_state.mem_user_count = 0
            # Auto-generate new group_id when clearing conversation
            import uuid
            st.session_state.group_id = f"chat-{uuid.uuid4()}"
            st.success(f"Conversation cleared. New group_id: {st.session_state.group_id}")
            st.rerun()

    # Info about current memory configuration
    current_n = st.session_state.get("ingest_every_n", 5)
    current_window = st.session_state.get("short_term_window", 5)
    
    st.markdown("### 🧠 Memory Configuration")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        if current_n == 1:
            st.info("💾 **Mid-term:** Direct save - Each turn → KG immediately")
        else:
            st.info(f"📝 **Mid-term:** Every {current_n} turns → Summarize → KG")
    with col_info2:
        st.info(f"⚡ **Short-term:** Last {current_window} turns kept as context")
    with col_info3:
        # Calculate total token usage and cost
        total_tokens = 0
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0
        for msg in st.session_state.chat_messages:
            if msg.get("token_usage"):
                usage = msg["token_usage"]
                total_tokens += usage["total_tokens"]
                total_prompt += usage["prompt_tokens"]
                total_completion += usage["completion_tokens"]
                total_cost += calculate_cost(
                    usage["prompt_tokens"], 
                    usage["completion_tokens"], 
                    usage["model"]
                )
        
        if total_tokens > 0:
            st.info(f"🔢 **Total Tokens:** {total_tokens:,}\n\n"
                   f"↘️ Out: {total_completion:,}| ↗️ In: {total_prompt:,}\n\n"
                   f"💰 **Cost:** ${total_cost:.4f}")
        else:
            st.info("🔢 **Total Tokens:** 0\n\nNo usage yet")
    
    # Group ID Management
    st.markdown("### 🔑 Conversation Group ID")
    
    col_gid1, col_gid2 = st.columns([3, 1])
    with col_gid1:
        # Text input to manually set group_id
        manual_group_id = st.text_input(
            "Group ID (để tiếp tục chat cũ, paste group_id cũ vào đây)",
            value=st.session_state.group_id,
            key="manual_group_id_input",
            help="Copy group_id từ conversation cũ để truy cập memories của nó"
        )
        
        # Update session state if user changes it
        if manual_group_id != st.session_state.group_id:
            # Warn if there are unsaved messages
            if st.session_state.chat_messages:
                st.warning("⚠️ **Warning:** Bạn đang có messages trong chat hiện tại!")
                st.info("💡 **Lưu ý:** Memories mới sẽ được lưu vào group_id mới này. Messages cũ (nếu chưa save) có thể bị mất.")
            
            st.session_state.group_id = manual_group_id
            st.session_state.mem_buffer = []
            st.session_state.mem_user_count = 0
            st.success(f"✓ Switched to group_id: {manual_group_id}")
            st.info("💡 Tất cả memories mới sẽ được lưu vào conversation này!")
    
    with col_gid2:
        # Button to generate new group_id
        if st.button("New Chat", key="regen_btn"):
            import uuid
            new_group_id = f"chat-{uuid.uuid4()}"
            st.session_state.group_id = new_group_id
            st.session_state.mem_buffer = []
            st.session_state.mem_user_count = 0
            st.session_state.chat_messages = []  # Also clear messages
            st.success(f"New conversation started!")
            st.rerun()
    
    # Copy button helper
    st.caption(f"💡 **Tip:** Copy group_id này để tiếp tục conversation sau khi reload page")
    st.code(st.session_state.group_id, language=None)
    
    # Export button
    st.markdown("### 💾 Export Conversation")
    if st.button("📥 Export to JSON", key="export_btn"):
        import json
        success, data = get_json(f"/export/{st.session_state.group_id}")
        if success:
            # Create download link
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"conversation_{st.session_state.group_id}.json",
                mime="application/json",
                key="download_json"
            )
            st.success(f"✓ Exported {data.get('entity_count', 0)} entities")
            st.info(f"💡 File contains all facts and entities from this conversation")
        else:
            st.error(f"Export failed: {data}")

    # OpenAI config
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("MODEL_NAME", "gpt-4.1-mini")

    # Process user input if exists (will be rendered at bottom)
    user_input = st.session_state.get("_pending_input", None)
    if user_input:
        # Clear pending input
        st.session_state.pop("_pending_input", None)
        # Get current config values
        chat_name = st.session_state.get("chat_name", "agent_chat")
        auto_save = st.session_state.get("auto_save", True)
        show_memories = st.session_state.get("show_memories", True)
        pause_saving = st.session_state.get("pause_saving", False)
        ingest_every_n_user_turns = st.session_state.get("ingest_every_n", 2)
        
        # Auto-determine if should summarize based on N
        summarize_to_memory = (ingest_every_n_user_turns > 1)
        
        # Append user message locally
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Save this user turn
        if auto_save and not pause_saving and not summarize_to_memory:
            # Optional: Translate to English for consistent entity summaries
            message_for_kg = user_input
            if st.session_state.get("normalize_language_to_english", False):
                # Translation would happen here
                # message_for_kg = translate_to_english(user_input)
                pass
            
            lines = [f"user: {message_for_kg}"]
            payload = {
                "name": chat_name,
                "messages": lines,
                "reference_time": datetime.utcnow().isoformat(),
                "source_description": "chat",
                "group_id": st.session_state.group_id,
            }
            post_json_timeout("/ingest/message", payload, timeout_sec=5)

        # Always add to grouped buffer when summarization is on
        if summarize_to_memory:
            st.session_state.mem_buffer.append(f"user: {user_input}")
            st.session_state.mem_user_count += 1
            
            # Debug: Show progress to KG ingest
            turns_remaining = ingest_every_n_user_turns - st.session_state.mem_user_count
            if show_memories:
                if turns_remaining > 0:
                    st.caption(f"🔄 Progress: {st.session_state.mem_user_count}/{ingest_every_n_user_turns} turns (ingest in {turns_remaining} more turns)")
                else:
                    st.caption(f"⚡ Triggering KG ingest now! ({st.session_state.mem_user_count}/{ingest_every_n_user_turns} turns reached)")

        # Decide whether to query KG (LLM-only decision when key is set)
        wants_kg = False
        results = []
        search_data = None

        if openai_key:
            # Ask LLM to decide if KG is needed
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                decision_prompt = format_decision_prompt(user_input)
                decision_config = PROMPT_CONFIG.get("decision", {})
                
                decision = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": decision_prompt}],
                    temperature=decision_config.get("temperature", 0.0),
                    max_tokens=decision_config.get("max_tokens", 10),
                )
                wants_kg = decision.choices[0].message.content.strip().upper().startswith("Y")
            except Exception:
                pass

        if wants_kg:
            ok_search, search_data = post_json_timeout("/search", {
                "query": user_input, 
                "focal_node_uuid": None,
                "group_id": st.session_state.group_id  # Filter by current conversation
            }, timeout_sec=10)
            if ok_search and isinstance(search_data, dict):
                results = search_data.get("results", []) or []
            if show_memories and search_data is not None:
                with st.expander("Related memories", expanded=False):
                    st.json(search_data)

        # Generate assistant reply (normal convo; ground with KG only if used)
        assistant_reply = None
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                
                # Short-term memory: Keep only last N turns (default: 5)
                # This creates a sliding window of recent context
                short_term_window = st.session_state.get("short_term_window", 5)
                all_messages = st.session_state.chat_messages
                
                if len(all_messages) > short_term_window * 2:  # *2 for user+assistant pairs
                    # Keep last N user-assistant pairs
                    history = all_messages[-(short_term_window * 2):]
                else:
                    history = all_messages
                
                history = [{"role": m["role"], "content": m["content"]} for m in history]
                
                # Build facts list from KG results (mid-term memory)
                top_facts = []
                if results:
                    for r in results[:8]:
                        if isinstance(r, dict):
                            txt = r.get("text") or r.get("fact") or r.get("name") or str(r)
                        else:
                            txt = str(r)
                        top_facts.append(txt)
                
                # Format system prompt with memories
                system_content = format_system_prompt(top_facts if top_facts else None)
                chat_config = PROMPT_CONFIG.get("chat", {})
                
                # Debug: Show current config and memory usage
                if show_memories:
                    st.info(f"🔧 LLM Config: max_tokens={chat_config.get('max_tokens', 5000)}, temp={chat_config.get('temperature', 0.9)}")
                    st.caption(f"⚡ Short-term: Using last {len(history)} messages ({len(history)//2} turns)")
                    
                    if top_facts:
                        st.caption(f"📝 Mid-term: {len(top_facts)} facts from KG")
                        with st.expander("📝 Facts injected into prompt", expanded=False):
                            for i, fact in enumerate(top_facts, 1):
                                st.markdown(f"{i}. {fact}")
                    else:
                        st.caption("ℹ️ No relevant memories found from KG")

                messages_for_llm = [{"role": "system", "content": system_content}] + history
                
                # Debug: Show full LLM input
                if show_memories:
                    with st.expander("🔍 Full LLM Input (Debug)", expanded=False):
                        st.markdown("### System Prompt")
                        st.code(system_content, language=None)
                        
                        st.markdown("### Conversation History (Short-term)")
                        for i, msg in enumerate(history, 1):
                            role_icon = "👤" if msg["role"] == "user" else "🤖"
                            st.markdown(f"**{i}. {role_icon} {msg['role'].upper()}:**")
                            st.code(msg["content"], language=None)
                        
                        st.markdown("### Complete Payload")
                        st.json(messages_for_llm, expanded=False)
                completion = client.chat.completions.create(
                    model=openai_model,
                    messages=messages_for_llm,
                    temperature=chat_config.get("temperature", 0.9),
                    max_tokens=chat_config.get("max_tokens", 5000),  # Fallback to 5000 if config fails -> # AI chỉ được trả lời tối đa 5000 tokens
                   # (khoảng 3750 chữ hoặc ~15-20 đoạn code)
                )
                assistant_reply = completion.choices[0].message.content.strip()
                
                # Capture token usage
                token_usage = None
                if hasattr(completion, 'usage') and completion.usage:
                    token_usage = {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens,
                        "model": openai_model
                    }
                
                # Check if response was truncated
                if completion.choices[0].finish_reason == "length":
                    st.warning("⚠️ Response was truncated due to max_tokens limit. Consider increasing it.")
            except Exception as e:
                st.warning(f"LLM error, using echo: {e}")
                assistant_reply = f"Bạn đã nói: {user_input}"
        else:
            assistant_reply = f"Bạn đã nói: {user_input}"
            token_usage = None

        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": assistant_reply,
            "token_usage": token_usage
        })
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
            
            # Display token usage if available
            if token_usage:
                cost = calculate_cost(
                    token_usage['prompt_tokens'], 
                    token_usage['completion_tokens'], 
                    token_usage['model']
                )
                st.caption(
                    f"🔢 Tokens: **{token_usage['completion_tokens']}** output | "
                    f"**{token_usage['prompt_tokens']}** input | "
                    f"**{token_usage['total_tokens']}** total | "
                    f"💰 **${cost:.4f}** ({token_usage['model']})"
                )

        # Save assistant turn if auto-save and not summarizing
        if auto_save and not pause_saving and not summarize_to_memory:
            payload = {
                "name": chat_name,
                "messages": [f"assistant: {assistant_reply}"],
                "reference_time": datetime.utcnow().isoformat(),
                "source_description": "chat",
                "group_id": st.session_state.group_id,
            }
            post_json_timeout("/ingest/message", payload, timeout_sec=5)

        # If summarizing, append assistant and maybe ingest summary
        if summarize_to_memory:
            st.session_state.mem_buffer.append(f"assistant: {assistant_reply}")
            # Ingest when reached N user turns
            if st.session_state.mem_user_count >= ingest_every_n_user_turns and not pause_saving:
                summary_text = None
                if openai_key:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=openai_key)
                        
                        # Get last N turns for summarization
                        recent_turns = st.session_state.mem_buffer[-(2*ingest_every_n_user_turns):]
                        sum_prompt = format_summarization_prompt(recent_turns)
                        sum_config = PROMPT_CONFIG.get("summarization", {})
                        
                        comp = client.chat.completions.create(
                            model=openai_model,
                            messages=[{"role": "user", "content": sum_prompt}],
                            temperature=sum_config.get("temperature", 0.2),
                            max_tokens=sum_config.get("max_tokens", 250),  # Increased for multiple facts
                        )
                        summary_text = comp.choices[0].message.content.strip()
                    except Exception:
                        summary_text = None
                if not summary_text:
                    # Fallback simple compression
                    last_lines = st.session_state.mem_buffer[-(2*ingest_every_n_user_turns):]
                    joined = "; ".join(last_lines)
                    summary_text = (joined[:180] + "…") if len(joined) > 180 else joined

                # Parse multiple facts from summary (one per line starting with "- ")
                facts = []
                if summary_text:
                    for line in summary_text.split('\n'):
                        line = line.strip()
                        if line.startswith('- '):
                            facts.append(line[2:].strip())
                        elif line and not line.startswith('**'):  # Skip markdown headers
                            facts.append(line)
                
                # Ingest each fact separately with importance filtering
                if facts:
                    from app.importance import get_scorer
                    scorer = get_scorer()
                    
                    ingested_count = 0
                    filtered_count = 0
                    
                    # Show progress
                    if show_memories:
                        st.info(f"📝 Ingesting {len(facts)} facts to Knowledge Graph...")
                        progress_bar = st.progress(0)
                    
                    for idx, fact in enumerate(facts):
                        if len(fact) > 10:  # Skip very short lines
                            # Check importance before ingesting
                            should_ingest, score_info = scorer.should_ingest(fact, threshold=0.3)
                            
                            if should_ingest:
                                payload = {
                                    "name": chat_name,
                                    "text": fact,
                                    "reference_time": datetime.utcnow().isoformat(),
                                    "source_description": "chat_summary",
                                    "group_id": st.session_state.group_id,
                                }
                                
                                # Retry logic with exponential backoff
                                max_retries = 2
                                success = False
                                for attempt in range(max_retries):
                                    success, response = post_json_timeout("/ingest/text", payload, timeout_sec=30)
                                    if success:
                                        break
                                    else:
                                        if attempt < max_retries - 1:
                                            import time
                                            wait_time = 2 ** attempt  # 1s, 2s
                                            if show_memories:
                                                st.caption(f"⏳ Retry {attempt+1}/{max_retries-1} in {wait_time}s...")
                                            time.sleep(wait_time)
                                
                                if success:
                                    ingested_count += 1
                                else:
                                    if show_memories:
                                        st.warning(f"⚠️ Failed to ingest fact after {max_retries} attempts: {fact[:50]}...")
                                    filtered_count += 1
                            else:
                                filtered_count += 1
                                if show_memories:
                                    st.caption(f"⚠️ Filtered low-importance fact: {fact[:50]}... (score: {score_info['score']})")
                        
                        # Update progress
                        if show_memories:
                            progress_bar.progress((idx + 1) / len(facts))
                    
                    if show_memories and filtered_count > 0:
                        st.info(f"💡 Filtered {filtered_count} low-importance facts, ingested {ingested_count}")
                else:
                    # Fallback: ingest as single blob
                    payload = {
                        "name": chat_name,
                        "text": summary_text,
                        "reference_time": datetime.utcnow().isoformat(),
                        "source_description": "chat_summary",
                        "group_id": st.session_state.group_id,
                    }
                    post_json_timeout("/ingest/text", payload, timeout_sec=6)
                # reset counter, keep buffer trimmed
                st.session_state.mem_user_count = 0

    # Manual persist full transcript
    if st.button("Save full conversation transcript"):
        lines = [f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages]
        payload = {
            "name": chat_name,
            "messages": lines,
            "reference_time": datetime.utcnow().isoformat(),
            "source_description": "chat",
            "group_id": st.session_state.group_id,
        }
        ok, data = post_json("/ingest/message", payload)
        if ok:
            st.success("Transcript saved")
        else:
            st.error("Failed to save transcript")
    
    # Render chat history in the container (at top)
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Display token usage for assistant messages if available
                if msg["role"] == "assistant" and msg.get("token_usage"):
                    usage = msg["token_usage"]
                    cost = calculate_cost(
                        usage['prompt_tokens'], 
                        usage['completion_tokens'], 
                        usage['model']
                    )
                    st.caption(
                        f"🔢 Tokens: **{usage['completion_tokens']}** output | "
                        f"**{usage['prompt_tokens']}** input | "
                        f"**{usage['total_tokens']}** total | "
                        f"💰 **${cost:.4f}** ({usage['model']})"
                    )
    
    # Chat input at the very bottom
    chat_input = st.chat_input("Type your message")
    if chat_input:
        st.session_state["_pending_input"] = chat_input
        st.rerun()

# ---------------------------- Ingest Tab ----------------------------
with tabs[1]:
    st.subheader("Ingest")

    with st.expander("Ingest Text", expanded=True):
        with st.form("ingest_text_form"):
            name = st.text_input("Name", value="sample-text")
            text = st.text_area("Text", height=150)
            ref_time_enabled = st.checkbox("Set reference time (ISO 8601)")
            reference_time = st.text_input("Reference time", value=datetime.utcnow().isoformat(), disabled=not ref_time_enabled)
            source_description = st.text_input("Source description", value="app")
            submitted = st.form_submit_button("Ingest Text")
            if submitted:
                payload = {
                    "name": name,
                    "text": text,
                    "reference_time": reference_time if ref_time_enabled else None,
                    "source_description": source_description,
                }
                ok, data = post_json("/ingest/text", payload)
                if ok:
                    st.success("Text ingested")
                    st.json(data)
                else:
                    st.error("Failed to ingest text")
                    st.json(data)

    with st.expander("Ingest Messages"):
        with st.form("ingest_messages_form"):
            name = st.text_input("Name", value="sample-chat")
            messages_raw = st.text_area(
                "Messages (one per line, e.g., 'user: hello')",
                height=150,
            )
            ref_time_enabled = st.checkbox("Set reference time (ISO 8601)", key="msg_ref")
            reference_time = st.text_input("Reference time", value=datetime.utcnow().isoformat(), disabled=not ref_time_enabled, key="msg_ref_input")
            source_description = st.text_input("Source description", value="chat", key="msg_src")
            submitted = st.form_submit_button("Ingest Messages")
            if submitted:
                messages: List[str] = [line.strip() for line in messages_raw.splitlines() if line.strip()]
                payload = {
                    "name": name,
                    "messages": messages,
                    "reference_time": reference_time if ref_time_enabled else None,
                    "source_description": source_description,
                }
                ok, data = post_json("/ingest/message", payload)
                if ok:
                    st.success("Messages ingested")
                    st.json(data)
                else:
                    st.error("Failed to ingest messages")
                    st.json(data)

    with st.expander("Ingest JSON"):
        with st.form("ingest_json_form"):
            name = st.text_input("Name", value="sample-json")
            json_text = st.text_area("JSON payload", value='{"key": "value"}', height=150)
            ref_time_enabled = st.checkbox("Set reference time (ISO 8601)", key="json_ref")
            reference_time = st.text_input("Reference time", value=datetime.utcnow().isoformat(), disabled=not ref_time_enabled, key="json_ref_input")
            source_description = st.text_input("Source description", value="json", key="json_src")
            submitted = st.form_submit_button("Ingest JSON")
            if submitted:
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                else:
                    payload = {
                        "name": name,
                        "json": parsed,  # alias for data
                        "reference_time": reference_time if ref_time_enabled else None,
                        "source_description": source_description,
                    }
                    ok, data = post_json("/ingest/json", payload)
                    if ok:
                        st.success("JSON ingested")
                        st.json(data)
                    else:
                        st.error("Failed to ingest JSON")
                        st.json(data)

    st.caption("Tip: Set environment variable MEMORY_LAYER_API to point to a different API base URL.")

# ---------------------------- Search Tab ----------------------------
with tabs[2]:
    st.subheader("Search")
    query = st.text_input("Query", value="hello", key="search_query")
    focal = st.text_input("Focal node UUID (optional)", key="search_focal")
    group_filter = st.text_input("Group ID (optional - filter by conversation)", key="search_group")
    if st.button("Run Search", key="search_button"):
        payload = {
            "query": query, 
            "focal_node_uuid": focal or None,
            "group_id": group_filter or None
        }
        ok, data = post_json("/search", payload)
        if ok:
            st.success("Search complete")
            st.json(data)
        else:
            st.error("Search failed")
            st.json(data)

# ---------------------------- Cache Tab ----------------------------
with tabs[3]:
    st.subheader("Cache Management")
    
    # Cache stats
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Get Cache Stats"):
            ok, data = post_json("/cache/stats", {})
            if ok:
                st.success("Cache stats retrieved")
                st.json(data)
            else:
                st.error("Failed to get cache stats")
                st.json(data)
    
    with col2:
        if st.button("Check Cache Health"):
            ok, data = post_json("/cache/health", {})
            if ok:
                if data.get("status") == "healthy":
                    st.success("Cache is healthy")
                else:
                    st.warning("Cache is empty")
                st.json(data)
            else:
                st.error("Failed to check cache health")
                st.json(data)
    
    # Cache management
    st.subheader("Cache Management Actions")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("Clear All Cache", type="secondary"):
            ok, data = post_json("/cache/clear", {})
            if ok:
                st.success("All cache cleared")
            else:
                st.error("Failed to clear cache")
                st.json(data)
    
    with col4:
        if st.button("Clear Search Cache", type="secondary"):
            ok, data = post_json("/cache/clear-search", {})
            if ok:
                st.success("Search cache cleared")
            else:
                st.error("Failed to clear search cache")
                st.json(data)
    
    with col5:
        node_uuid = st.text_input("Node UUID to clear", key="clear_node_uuid")
        if st.button("Clear Node Cache", type="secondary"):
            if node_uuid:
                ok, data = post_json(f"/cache/clear-node/{node_uuid}", {})
                if ok:
                    st.success(f"Cache for node {node_uuid} cleared")
                else:
                    st.error("Failed to clear node cache")
                    st.json(data)
            else:
                st.warning("Please enter a node UUID")
    
    # Cache information
    st.subheader("Cache Information")
    st.info("""
    **Cache Types:**
    - **Search Cache**: Cached search results (TTL: 30 minutes)
    - **Node Cache**: Cached node data (TTL: 1 hour)
    - **Connection Cache**: Cached node connections (TTL: 30 minutes)
    
    **Cache Invalidation:**
    - Search cache is automatically cleared when new data is ingested
    - Node cache can be manually cleared for specific nodes
    - All caches can be cleared manually for maintenance
    """)

# ---------------------------- Debug Tab ----------------------------
with tabs[4]:
    st.subheader("Debug Tools")
    
    st.markdown("### Check Episodes by Group ID")
    debug_group_id = st.text_input(
        "Enter Group ID to debug", 
        value=st.session_state.get("group_id", ""),
        key="debug_group_id"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Show Episodes", key="debug_episodes"):
            if debug_group_id:
                try:
                    import requests
                    base = get_api_base_url()
                    resp = requests.get(f"{base}/debug/episodes/{debug_group_id}", timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    st.success(f"Found {data.get('count', 0)} episodes")
                    st.json(data)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a group_id")
    
    with col2:
        if st.button("Show Edges/Facts", key="debug_edges"):
            if debug_group_id:
                try:
                    import requests
                    base = get_api_base_url()
                    resp = requests.get(f"{base}/debug/edges/{debug_group_id}", timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    st.success(f"Found {data.get('count', 0)} edges")
                    st.json(data)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a group_id")
    
    st.markdown("---")
    st.markdown("### Current Session Info")
    st.json({
        "current_group_id": st.session_state.get("group_id", "Not set"),
        "chat_messages_count": len(st.session_state.get("chat_messages", [])),
        "mem_buffer_size": len(st.session_state.get("mem_buffer", [])),
        "mem_user_count": st.session_state.get("mem_user_count", 0)
    })


