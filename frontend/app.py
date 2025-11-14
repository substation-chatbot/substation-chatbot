import streamlit as st
import requests

# ==========================
# CONFIG

API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="⚡ Substation Maintenance Chatbot", layout="wide")

# ==========================
# CSS Styling

st.markdown("""
    <style>
    .chat-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 12px;
        max-height: 75vh;
        overflow-y: auto;
        margin-bottom: 10px;
    }
    .user-msg {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 8px 0;
        text-align: right;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
        color: black;
    }
    .bot-msg {
        background-color: #ffffff;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 8px 0;
        text-align: left;
        width: fit-content;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
        color: black;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Intelligent Substation Maintenance Chatbot")

# ==========================
# SESSION STATE INIT

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if len(st.session_state.chat_history) > 5:  # keep only last 5 messages
    st.session_state.chat_history = st.session_state.chat_history[-5:]

# ==========================
# Upload Section

with st.expander("📂 Upload New Maintenance Documents", expanded=False):
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        with st.spinner("Uploading and embedding..."):
            res = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
            )
        result = res.json()
        if "message" in result:
            st.success(result["message"])
        else:
            st.error(result.get("error", "Unknown error occurred."))

# ==========================
# Chat Display Section

st.markdown("### 💬 Chat Window")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Chat Input

query = st.chat_input("Type your question about substation maintenance...")

if query:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/ask", data={"query": query})
        answer = response.json().get("answer", "⚠️ No response from the bot.")

    # Append bot response
    st.session_state.chat_history.append({"role": "bot", "content": answer})

    # Limit history to last 5 messages
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]

    st.rerun()


# ==========================
# Footer

st.caption("🤖  Developed for Substation Maintenance Assistance")

