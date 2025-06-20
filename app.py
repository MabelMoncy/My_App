import os
import streamlit as st
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import hashlib
import pickle
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Configure ChromaDB
embedding_function = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)
chroma_client = chromadb.PersistentClient(path="chromadb_store")
collection = chroma_client.get_or_create_collection(
    name="pdf_chunks",
    embedding_function=embedding_function
)

# Streamlit UI config
st.set_page_config(page_title="Alpy ATP bot")

st.title("Algorithm Thinking - Python")

st.sidebar.title("About the Subject")
st.sidebar.markdown("""
'Algorithm Thinking with Python' is a newly introduced subject in the 2024 KTU B.Tech syllabus under the 2024 scheme.
This course is designed to strengthen logical thinking skills while introducing students to the fundamentals of Python programming.
It serves as a foundational step toward problem-solving using algorithms in real-world scenarios.
""")
st.sidebar.title("About This App.")
st.sidebar.markdown("""
This web application is built to help students engage more deeply with the subject by allowing them to ask questions directly from the preloaded official textbook of Algorithm Thinking with Python.
The AI-powered system retrieves relevant content from the textbook and generates clear, contextual answers. It's your personal study assistant for better understanding core concepts, code examples, and logic-based problems — all from one place.
""")
# Inject custom CSS for better mobile experience
st.markdown("""
<style>

/* 1. Raise input box and prevent it from hiding behind bottom bezel */
.st-emotion-cache-13ln4jf {
   
    bottom: 150px;  /* Adjust height above mobile gesture bar */
    left: 0;
    right: 0;
    background-color: white;
    padding: 10px 12px;
    z-index: 999;
    border-top: 1px solid #ddd;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
}

/* 2. Add bottom padding to the main container to prevent overlap */
.st-emotion-cache-z5fcl4 {
    padding-bottom: 140px !important;  /* Leave space for fixed input box */
}

/* 3. Align answer content closer to the left */
.stMarkdown {
    text-align: left !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
}

/* 4. Make heading smaller on mobile screens */
@media screen and (max-width: 480px) {
    h1 {
        font-size: 22px !important;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
}

/* Optional: improve chat message bubbles on small screens */
.stChatMessage {
    margin-left: 4px !important;
    margin-right: 4px !important;
    padding: 6px !important;
}

</style>
""", unsafe_allow_html=True)



def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if (
            len(line) < 30 or
            re.match(r"^Page\s*\d+", line, re.I) or
            re.search(r"Author|Copyright", line, re.I)
        ):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

@st.cache_data
def load_pdf_chunks(pdf_path):
    with open("ATP_Split.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(raw_text)
    return chunks

def get_pdf_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def save_hash(current_hash, filename="pdf_hash.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(current_hash, f)

def load_hash(filename="pdf_hash.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def store_chunks_if_pdf_changed(chunks, pdf_path):
    clean_chunks = [str(chunk).strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    ids = [str(i) for i in range(len(clean_chunks))]
    collection.add(documents=clean_chunks, ids=ids)

def get_top_chunks(query, top_k=7):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    top_chunks = results.get("documents", [[]])[0]
    return [str(doc) for doc in top_chunks if isinstance(doc, str) and doc.strip()]

def answer_question(query):
    # If it's a vague query, reuse the last context
    vague_prompts = ["give an example", "explain more", "what about it", "why", "how","it"]
    if any(query.lower().startswith(x) for x in vague_prompts):
        context = st.session_state.get("last_context", "")
    else:
        top_chunks = get_top_chunks(query)
        context = "\n---\n".join(top_chunks)
        # Save this context for vague follow-up
        st.session_state.last_context = context

    messages = [{"role": "system", "content": "You are a teacher who answers questions from a textbook for students affiliated with APJ Abdul Kalam Technological University."}]
    
    # Chat history
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    prompt = f"Answer the question using the context below.\nContext:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )

    def stream_generator():
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return stream_generator()



# Main flow
pdf_path = "ATP_Split.txt"
with st.spinner("Checking PDF Text book status..."):
    current_hash = get_pdf_hash(pdf_path)
    previous_hash = load_hash()

    if current_hash != previous_hash:
        st.info("🔄 Loading text book content and preparing...")
        chunks = load_pdf_chunks(pdf_path)
        store_chunks_if_pdf_changed(chunks, pdf_path)
        save_hash(current_hash)
        st.success("📚 Text Book has been successfully processed and stored.")
    else:
        st.success("✅ Text Book already processed and up-to-date.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Wanna ask anything from the text book..?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # ✅ Spinner ends before streaming
        with st.spinner("Thinking..."):
            generator = answer_question(user_query)

        for chunk in generator:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
