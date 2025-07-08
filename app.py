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
from datetime import datetime

try:
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
    st.set_page_config(page_title="AI PDF Q&A")
    st.title("Algorithm Thinking - Python")

    st.sidebar.title("About the Subject")
    st.sidebar.markdown("""
    'Algorithm Thinking with Python' is a newly introduced subject in the 2024 KTU B.Tech syllabus under the 2024 scheme.
    This course is designed to strengthen logical thinking skills while introducing students to the fundamentals of Python programming.
    It serves as a foundational step toward problem-solving using algorithms in real-world scenarios.
    """)
    st.sidebar.title("About This App")
    st.sidebar.markdown("""
    This web application is built to help students engage more deeply with the subject by allowing them to ask questions directly from the preloaded official textbook of Algorithm Thinking with Python.
    The AI-powered system retrieves relevant content from the textbook and generates clear, contextual answers. It's your personal study assistant for better understanding core concepts, code examples, and logic-based problems — all from one place.
    """)
    st.markdown("""
    <style>
    /* 1. Raise input box and prevent it from hiding behind bottom bezel */
    .st-emotion-cache-13ln4jf {
        bottom: 150px;
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
        padding-bottom: 140px !important;
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
                len(line) < 30
                or re.match(r"^Page\s*\d+", line, re.I)
                or re.search(r"Author|Copyright", line, re.I)
            ):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @st.cache_data
    def load_pdf_chunks(pdf_path):
        with open(pdf_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            pages = raw_text.split("[Page ")[1:]
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
        collection.add(
            documents=clean_chunks,
            ids=ids
        )

    def get_top_chunks(query, top_k=7):
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        top_chunks = results.get("documents", [[]])[0]
        return [str(doc) for doc in top_chunks if isinstance(doc, str) and doc.strip()]

    def answer_question(query):
        try:
            top_chunks = get_top_chunks(query)
            context = "\n---\n".join(top_chunks)
            
            # System prompt with clear instructions
            system_prompt = (
                "You are a teacher answering questions based on the 'Algorithm Thinking with Python' textbook for B.Tech students "
                "affiliated with APJ Abdul Kalam Technological University. Provide accurate, concise, and context-aware answers "
                "using the provided PDF content and the conversation history. Ensure responses are relevant to the subject and "
                "helpful for students learning Python and algorithmic thinking."
            )
            
            # Build messages with history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Limit history to last 10 messages to avoid excessive context
            history_limit = 10
            history = st.session_state.messages[-history_limit:] if st.session_state.messages else []
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Construct the prompt with context and query
            prompt = f"Context from textbook:\n{context}\n\nQuestion: {query}\nAnswer:"
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
                    elif chunk.choices[0].finish_reason:
                        break
            
            return stream_generator()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return iter(["An error occurred while generating the response. Please try again."])

    # Main flow
    pdf_path = "ATP_split.txt"
    with st.spinner("Checking PDF status..."):
        current_hash = get_pdf_hash(pdf_path)
        previous_hash = load_hash()
        if current_hash != previous_hash:
            st.info("🔄 Loading PDF content and preparing...")
            chunks = load_pdf_chunks(pdf_path)
            st.info(f"✅ PDF loaded successfully.")
            store_chunks_if_pdf_changed(chunks, pdf_path)
            save_hash(current_hash)
            st.success("📚 PDF has been successfully processed and stored.")
        else:
            st.success("✅ PDF already processed and up-to-date.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

    # Display past messages with timestamps
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            timestamp = message.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.markdown(f"**[{timestamp}]** {message['content']}")

    # Handle user input
    if user_query := st.chat_input("What do you want to know?"):
        # Add user's message to history with timestamp
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Assistant response (streamed)
        with st.chat_message("assistant"):
            full_response = ""
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                for chunk in answer_question(user_query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            # Save assistant response to history with timestamp
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
except:
    st.error(f"Oops! Something went wrong. Please try again.")
    st.stop()
