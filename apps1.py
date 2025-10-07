import streamlit as st
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, qa_model

summarizer, qa_model = load_models()

# ---------- Helper Function ----------
def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')

    sentences = inp_str.split('<eos>')
    chunks, current_chunk = [], 0
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))
    return [' '.join(chunk) for chunk in chunks]

# ---------- Page Config ----------
st.set_page_config(page_title="Creative Summarizer & QA", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #89f7fe, #66a6ff);
    font-family: 'Segoe UI', sans-serif;
    color: #1e1e1e;
}

h1 {
    text-align: center;
    font-size: 3em;
    font-weight: 900;
    color: #1e3c72;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    margin-bottom: 0.2em;
}

.stTextArea textarea {
    border-radius: 15px !important;
    border: none !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15) !important;
    font-size: 16px !important;
    padding: 12px;
    background-color: #ffffff !important;
    color: #1e1e1e !important;
}

.stButton>button {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    color: white !important;
    font-weight: bold;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    font-size: 18px !important;
    width: 100%;
    transition: all 0.3s ease-in-out;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #2a5298, #1e3c72);
    transform: scale(1.05);
}

.result-box {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
    color: #1e1e1e;
    font-weight: bold;
    font-size: 17px;
    margin-top: 25px;
    animation: fadeIn 0.6s ease-in-out;
}

.emoji {
    font-size: 2em;
    color: #4a90e2;
    vertical-align: middle;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ---------- UI ----------
st.markdown("<h1>Creative Text Summarizer & QA</h1>", unsafe_allow_html=True)

# Text input
text_input = st.text_area("📝 Paste your text here:", height=220, placeholder="Enter or paste your article...")

# Buttons
col1, col2 = st.columns([1,1])
with col1:
    summarize_btn = st.button("✨ Summarize")
with col2:
    question_btn = st.button("❓ Ask Question")

# ---------- Summarization ----------
if summarize_btn:
    if text_input.strip() == "":
        st.warning("Please enter text to summarize!")
    else:
        with st.spinner("Generating summary..."):
            chunks = generate_chunks(text_input)
            res = summarizer(chunks, max_length=120, min_length=30)
            summary_text = ' '.join([summ['summary_text'] for summ in res])
        st.markdown(f"<div class='result-box'><span class='emoji'>✂️</span> Summary:<br><br>{summary_text}</div>", unsafe_allow_html=True)
        st.session_state['summary'] = summary_text

# ---------- Question Answering ----------
if question_btn:
    if 'summary' not in st.session_state or st.session_state['summary'].strip() == "":
        st.warning("Please summarize text first!")
    else:
        question = st.text_input("💬 Type your question:")
        if question:
            with st.spinner("Finding answer..."):
                answer = qa_model(question=question, context=st.session_state['summary'])
            st.markdown(f"<div class='result-box'><span class='emoji'>🧩</span> Answer:<br><br>{answer['answer']}</div>", unsafe_allow_html=True)
