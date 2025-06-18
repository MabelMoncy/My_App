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
st.set_page_config(page_title="AI PDF Q&A", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background:#aaaaaa;
        color: Black;
        font-family: Arial;
        font-size: 30px;
    }

    h1,h3{
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("📄 Algorithm Thinking With Python")
st.subheader("AI-Powered PDF Question Answering")
st.markdown("📘About the Subject\n\n 'Algorithm Thinking with Python' is a newly introduced subject in the 2024 KTU B.Tech syllabus under the 2024 scheme. This course is designed to strengthen logical thinking skills while introducing students to the fundamentals of Python programming. It serves as a foundational step toward problem-solving using algorithms in real-world scenarios.\n\n📔 About This App.\n\n This web application is built to help students engage more deeply with the subject by allowing them to ask questions directly from the preloaded official textbook of Algorithm Thinking with Python. The AI-powered system retrieves relevant content from the textbook and generates clear, contextual answers. It's your personal study assistant for better understanding core concepts, code examples, and logic-based problems — all from one place.")
st.markdown("Ask questions based on our preloaded Text Book of Algorithm thinking with python.")


def clean_text(text):
    # Remove boilerplate like author names, table of contents, headers, page numbers
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip lines that are:
        if (
            len(line) < 30                # Too short, likely not useful
            or re.match(r"^Page\s*\d+", line, re.I)  # Matches "Page 1", etc.
            or re.search(r"Author|Copyright", line, re.I)
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# Load and split PDF into chunks
@st.cache_data
def load_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            cleaned = clean_text(text)
            raw_text += cleaned + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(raw_text)
    return chunks

# Store chunks to ChromaDB if not already stored
# Utility: Get hash of the PDF file
def get_pdf_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Utility: Save hash to a file
def save_hash(current_hash, filename="pdf_hash.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(current_hash, f)

# Utility: Load previous hash
def load_hash(filename="pdf_hash.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# Updated: store chunks only if PDF changed
def store_chunks_if_pdf_changed(chunks, pdf_path):
    current_hash = get_pdf_hash(pdf_path)
    previous_hash = load_hash()

    if current_hash != previous_hash:
        # Fetch existing documents
        existing = collection.get()
        if existing and "ids" in existing and existing["ids"]:
            collection.delete(ids=existing["ids"])

        # Store new chunks
        collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": f"page_{i}"} for i in range(len(chunks))]
        )
        save_hash(current_hash)


# Get top relevant chunks
def get_top_chunks(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]  # list of top chunks

# Generate answer using GPT-4
def answer_question(query):
    top_chunks = get_top_chunks(query)
    context = "\n---\n".join(top_chunks)
    prompt = f"Answer the question using the context below.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a teacher who answers questions based on stored pdf ATP.pdf to B tech students in kerala affliated to APJ abdhul kalam university"},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    # Stream generator
    def stream_generator():
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return stream_generator()
    

# Main flow
with st.spinner("Loading PDF content and preparing..."):
    chunks = load_pdf_chunks("pdfs/ATP_split.pdf")
    store_chunks_if_pdf_changed(chunks,"pdfs/ATP_split.pdf")

user_query = st.text_input("",placeholder="e.g., What is Algorithm Thinking with python?")

if user_query:
    with st.spinner("Thinking..."):
        response = answer_question(user_query)
        st.markdown("## Answer:")
        st.write(response)

   
