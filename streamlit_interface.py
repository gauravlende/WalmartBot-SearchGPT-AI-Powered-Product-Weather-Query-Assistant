import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Walmart SalesBot & SearchGPT",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling
st.markdown("""
<style>
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
        background: transparent;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .source-card {
        background: rgba(0, 113, 220, 0.08);
        border-left: 4px solid #0071dc;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.9em;
    }
    .latency-badge {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 4px;
    }
    .suggestion-btn {
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .stApp {
        background-image: url('file:///C:/Users/hp/.gemini/antigravity-ide/brain/a5d99642-02a3-4f54-b1b7-0205e1845194/modern_background_1788431650850.jpg') !important;
        background-size: cover !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }
    .stApp > .main {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = "WalmartBot"

def check_server_status():
    try:
        r = requests.get("http://localhost:5000/", timeout=1.5)
        return r.status_code == 200
    except Exception:
        pass
# Sidebar
with st.sidebar:
    st.title("⚙️ Bot Settings")
    
    selected_bot = st.radio(
        "Choose AI Assistant:",
        ("WalmartBot 🛒", "SearchGPT 🔍"),
        index=0 if "WalmartBot" in st.session_state.selected_bot else 1
    )
    bot_type = "WalmartBot" if "WalmartBot" in selected_bot else "SearchGPT"
    
    if bot_type != st.session_state.selected_bot:
        st.session_state.selected_bot = bot_type
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        **About Assistants:**
        - **WalmartBot**: Instant grocery catalog search across 7,000+ Walmart items with prices & direct URLs.
        - **SearchGPT**: Fast web search, live weather, and witty conversational AI.
        """
    )

# Main Header
if st.session_state.selected_bot == "WalmartBot":
    st.title("🛒 Walmart SalesBot AI")
    st.caption("Ask for any grocery products, prices, or recommendations (e.g., *'Do you have organic tomatoes?'*)")
else:
    st.title("🔍 SearchGPT Assistant")
    st.caption("Ask anything from real-time web facts to weather and jokes!")

# Quick suggestion buttons
cols = st.columns(4)
suggestions = {
    "WalmartBot": [
        "🍅 Organic cherry tomatoes",
        "🥛 Organic milk and cheese",
        "🍪 Ben & Jerry's Ice Cream",
        "☕ Sweet cream coffee creamer"
    ],
    "SearchGPT": [
        "☀️ Weather in New York",
        "⚽ Latest sports headlines",
        "🎭 Tell me a funny joke",
        "🚀 Latest tech news"
    ]
}

suggested_query = None
for i, sugg in enumerate(suggestions[st.session_state.selected_bot]):
    if cols[i].button(sugg, key=f"sugg_{i}", use_container_width=True):
        suggested_query = sugg

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else ("🛒" if st.session_state.selected_bot == "WalmartBot" else "🤖")):
        st.markdown(msg["content"])
        if msg.get("sources"):
            for src in msg["sources"]:
                st.markdown(f"📚 **Source**: [{src}]({src})" if src.startswith("http") else f"📚 **Source**: {src}")
        if msg.get("latency"):
            st.markdown(f"<div class='latency-badge'>⚡ Responded in {msg['latency']}s</div>", unsafe_allow_html=True)

# Chat Input & Processing
user_input = st.chat_input("Type your message here...") or suggested_query

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Call Backend API
    with st.chat_message("assistant", avatar="🛒" if st.session_state.selected_bot == "WalmartBot" else "🤖"):
        with st.spinner(f"Getting response from {st.session_state.selected_bot}..."):
            t_start = time.time()
            bot_text = ""
            sources = []
            is_error = False

            try:
                if st.session_state.selected_bot == "WalmartBot":
                    endpoint = "http://localhost:5000/walmartbot"
                    resp = requests.post(endpoint, json={"messages": [user_input]}, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    bot_text = data.get("messages", ["No response"])[0]
                    sources = data.get("sources", [])
                    is_error = data.get("error", False)
                else:
                    endpoint = "http://localhost:5000/searchgpt"
                    resp = requests.post(endpoint, json={"text": user_input}, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    bot_text = data.get("messages", ["No response"])[0]
                    is_error = data.get("error", False)

            except requests.exceptions.Timeout:
                bot_text = "⚠️ Request timed out. The backend took too long to respond."
                is_error = True
            except requests.exceptions.ConnectionError:
                bot_text = "⚠️ Cannot connect to backend server. Please run: `python run_api.py`"
                is_error = True
            except Exception as e:
                bot_text = f"⚠️ Error: {str(e)}"
                is_error = True

            latency = round(time.time() - t_start, 2)

            # Display response
            if is_error:
                st.error(bot_text)
            else:
                st.markdown(bot_text)

            if sources:
                st.markdown("#### 🔗 Product Sources:")
                for src in sources:
                    st.markdown(f"- {src}")

            st.markdown(f"<div class='latency-badge'>⚡ Responded in {latency}s</div>", unsafe_allow_html=True)

            # Save in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_text,
                "sources": sources,
                "latency": latency
            })
