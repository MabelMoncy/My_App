import os
import streamlit as st
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI as OpAi
from dotenv import load_dotenv
import hashlib
import pickle
import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from contextlib import contextmanager
import json
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ATPBotConfig:
    """Configuration class for the ATP Bot application."""
    
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pdf_path = "ATP_Split.txt"
        self.hash_file = "pdf_hash.pkl"
        self.chromadb_path = "chromadb_store"
        self.collection_name = "pdf_chunks"
        self.embedding_model = "text-embedding-3-small"  # Better embedding model
        self.chat_model = "gpt-4o-mini"
        self.chunk_size = 800  # Increased for better context
        self.chunk_overlap = 200  # Increased overlap for continuity
        self.top_k_results = 15  # More results for better accuracy
        self.min_similarity_score = 0.65  # Lower threshold for more inclusive search
        self.max_chat_history = 10  # Limit chat history for context window
        self.context_window_size = 4000  # Characters for context
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

class ConversationMemory:
    """Manages conversation memory and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        
    def get_conversation_context(self, messages: List[Dict]) -> str:
        """Extract relevant context from conversation history."""
        if not messages:
            return ""
        
        # Get recent conversation
        recent_messages = messages[-self.max_history:]
        
        # Build conversation summary
        context_parts = []
        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"][:500]  # Truncate long messages
            if role == "user":
                context_parts.append(f"Student asked: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant answered: {content}")
        
        return "\n".join(context_parts)
    
    def extract_key_topics(self, messages: List[Dict]) -> List[str]:
        """Extract key topics from conversation history."""
        topics = []
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"].lower()
                # Extract potential topics (simple keyword extraction)
                words = re.findall(r'\b[a-zA-Z]{4,}\b', content)
                topics.extend(words)
        
        # Return unique topics, most recent first
        return list(dict.fromkeys(topics))[:10]

class DocumentProcessor:
    """Enhanced document processing with better chunking strategies."""
    
    def __init__(self, config: ATPBotConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True
        )
    
    @st.cache_data
    def load_pdf_chunks(_self, pdf_path: str) -> Tuple[List[str], List[Dict]]:
        """Load and split PDF content into chunks with metadata."""
        try:
            with open(pdf_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            if not raw_text.strip():
                raise ValueError("PDF file is empty or contains no readable text")
            
            # Split into chunks
            chunks = _self.text_splitter.split_text(raw_text)
            
            # Create chunks with metadata
            processed_chunks = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    processed_chunks.append(chunk.strip())
                    
                    # Extract metadata
                    metadata = {
                        'chunk_id': i,
                        'length': len(chunk),
                        'has_code': bool(re.search(r'(def |class |import |for |if |while )', chunk)),
                        'has_example': 'example' in chunk.lower(),
                        'has_algorithm': any(word in chunk.lower() for word in ['algorithm', 'step', 'procedure']),
                        'section_type': _self._identify_section_type(chunk)
                    }
                    chunk_metadata.append(metadata)
            
            return processed_chunks, chunk_metadata
        
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading PDF chunks: {str(e)}")
            raise
    
    def _identify_section_type(self, chunk: str) -> str:
        """Identify the type of content in a chunk."""
        chunk_lower = chunk.lower()
        
        if any(word in chunk_lower for word in ['chapter', 'unit', 'lesson']):
            return 'header'
        elif re.search(r'(def |class |import |>>>)', chunk):
            return 'code'
        elif any(word in chunk_lower for word in ['example', 'consider', 'let us']):
            return 'example'
        elif any(word in chunk_lower for word in ['algorithm', 'step', 'procedure']):
            return 'algorithm'
        elif chunk.count('\n') <= 2 and len(chunk) < 200:
            return 'definition'
        else:
            return 'content'
    
    def get_pdf_hash(self, pdf_path: str) -> str:
        """Generate MD5 hash of the PDF file."""
        try:
            with open(pdf_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating PDF hash: {str(e)}")
            raise
    
    def save_hash(self, current_hash: str, filename: str = None) -> None:
        """Save hash to file."""
        filename = filename or self.config.hash_file
        try:
            with open(filename, "wb") as f:
                pickle.dump(current_hash, f)
        except Exception as e:
            logger.error(f"Error saving hash: {str(e)}")
            raise
    
    def load_hash(self, filename: str = None) -> Optional[str]:
        """Load hash from file."""
        filename = filename or self.config.hash_file
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading hash: {str(e)}")
        return None

class EnhancedVectorStore:
    """Enhanced vector store with better search capabilities."""
    
    def __init__(self, config: ATPBotConfig):
        self.config = config
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=config.openai_api_key,
            model_name=config.embedding_model
        )
        self.chroma_client = chromadb.PersistentClient(path=config.chromadb_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.embedding_function
        )
        self.chunk_metadata = []
    
    def store_chunks(self, chunks: List[str], metadata: List[Dict] = None) -> None:
        """Store text chunks with metadata in vector database."""
        try:
            clean_chunks = [chunk for chunk in chunks if chunk.strip()]
            if not clean_chunks:
                raise ValueError("No valid chunks to store")
            
            # Store metadata
            if metadata:
                self.chunk_metadata = metadata
            
            ids = [str(i) for i in range(len(clean_chunks))]
            
            # Clear existing collection
            try:
                existing_ids = self.collection.get()['ids']
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
            except:
                pass
            
            # Add chunks with metadata
            metadatas = metadata or [{'chunk_id': i} for i in range(len(clean_chunks))]
            self.collection.add(
                documents=clean_chunks, 
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(clean_chunks)} chunks in vector database")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise
    
    def search_similar_chunks(self, query: str, conversation_topics: List[str] = None) -> Tuple[List[str], List[Dict]]:
        """Enhanced search with conversation context."""
        try:
            if not query.strip():
                return [], []
            
            # Enhance query with conversation context
            enhanced_query = self._enhance_query_with_context(query, conversation_topics)
            
            # Primary search
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=self.config.top_k_results,
                include=["documents", "distances", "metadatas"]
            )
            
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            # Score and rank results
            scored_results = []
            for doc, distance, metadata in zip(documents, distances, metadatas):
                if doc.strip():
                    # Convert distance to similarity
                    similarity = max(0, 1 - distance)
                    
                    # Apply content-based scoring
                    content_score = self._calculate_content_score(doc, query, metadata)
                    
                    # Combined score
                    final_score = (similarity * 0.7) + (content_score * 0.3)
                    
                    scored_results.append({
                        'document': doc,
                        'score': final_score,
                        'metadata': metadata,
                        'similarity': similarity
                    })
            
            # Sort by score and filter
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Filter by minimum threshold
            filtered_results = [
                r for r in scored_results 
                if r['score'] >= self.config.min_similarity_score
            ]
            
            # If no results meet threshold, return top 3
            if not filtered_results:
                filtered_results = scored_results[:3]
            
            # Diversify results (avoid too many similar chunks)
            diversified_results = self._diversify_results(filtered_results[:8])
            
            documents = [r['document'] for r in diversified_results]
            metadata_list = [r['metadata'] for r in diversified_results]
            
            return documents, metadata_list
            
        except Exception as e:
            logger.error(f"Error searching chunks: {str(e)}")
            return [], []
    
    def _enhance_query_with_context(self, query: str, topics: List[str] = None) -> str:
        """Enhance query with conversation context."""
        enhanced_query = query
        
        if topics:
            # Add relevant topics to query context
            relevant_topics = [t for t in topics if len(t) > 3][:3]
            if relevant_topics:
                enhanced_query += f" Context: {' '.join(relevant_topics)}"
        
        return enhanced_query
    
    def _calculate_content_score(self, document: str, query: str, metadata: Dict) -> float:
        """Calculate content-based relevance score."""
        score = 0.0
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Exact keyword matches
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        doc_words = set(re.findall(r'\b\w+\b', doc_lower))
        keyword_overlap = len(query_words.intersection(doc_words))
        score += (keyword_overlap / len(query_words)) * 0.3 if query_words else 0
        
        # Content type preferences
        if 'code' in query_lower and metadata.get('has_code'):
            score += 0.2
        if 'example' in query_lower and metadata.get('has_example'):
            score += 0.2
        if 'algorithm' in query_lower and metadata.get('has_algorithm'):
            score += 0.2
        
        # Length preference (not too short, not too long)
        doc_length = metadata.get('length', 0)
        if 200 <= doc_length <= 800:
            score += 0.1
        
        return min(score, 1.0)
    
    def _diversify_results(self, results: List[Dict]) -> List[Dict]:
        """Diversify results to avoid redundancy."""
        diversified = []
        seen_content = set()
        
        for result in results:
            doc = result['document']
            # Use first 100 characters as similarity key
            content_key = doc[:100].lower()
            
            # Check if content is too similar to already selected
            is_similar = any(
                self._text_similarity(content_key, seen) > 0.8 
                for seen in seen_content
            )
            
            if not is_similar:
                diversified.append(result)
                seen_content.add(content_key)
            
            if len(diversified) >= 6:  # Limit results
                break
        
        return diversified
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

class EnhancedChatBot:
    """Enhanced chatbot with conversation memory and better responses."""
    
    def __init__(self, config: ATPBotConfig, vector_store: EnhancedVectorStore):
        self.config = config
        self.vector_store = vector_store
        self.client = OpAi(api_key=config.openai_api_key)
        self.memory = ConversationMemory(config.max_chat_history)
        
        self.vague_prompts = {
            "give an example", "explain more", "what about it", "why", "how", 
            "it", "tell me more", "elaborate", "continue", "go on", "what else",
            "can you", "show me", "help", "please", "okay", "yes", "no"
        }
    
    def is_vague_query(self, query: str) -> bool:
        """Enhanced vague query detection."""
        query_lower = query.lower().strip()
        
        # Check for vague starters
        vague_starters = any(query_lower.startswith(prompt) for prompt in self.vague_prompts)
        
        # Check for very short queries
        short_query = len(query_lower.split()) <= 2
        
        # Check for question words without context
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        is_bare_question = any(query_lower == word or query_lower == word + '?' for word in question_words)
        
        return vague_starters or (short_query and is_bare_question)
    
    def get_enhanced_context(self, query: str, messages: List[Dict]) -> Tuple[str, str]:
        """Get enhanced context including conversation history."""
        # Get conversation context
        conversation_context = self.memory.get_conversation_context(messages)
        topics = self.memory.extract_key_topics(messages)
        
        if self.is_vague_query(query):
            # For vague queries, prioritize recent context
            textbook_context = st.session_state.get("last_context", "")
            if not textbook_context and messages:
                # If no stored context, search based on recent conversation
                recent_queries = [msg["content"] for msg in messages[-3:] if msg["role"] == "user"]
                combined_query = " ".join(recent_queries) + " " + query
                chunks, _ = self.vector_store.search_similar_chunks(combined_query, topics)
                textbook_context = "\n---\n".join(chunks)
        else:
            # For specific queries, search textbook
            chunks, metadata = self.vector_store.search_similar_chunks(query, topics)
            textbook_context = "\n---\n".join(chunks)
            # Store for potential follow-up questions
            st.session_state.last_context = textbook_context
        
        return textbook_context, conversation_context
    
    def create_enhanced_system_message(self, has_conversation_history: bool = False) -> str:
        """Create enhanced system message with conversation awareness."""
        base_message = (
            "You are an expert teaching assistant for 'Algorithm Thinking with Python' "
            "at APJ Abdul Kalam Technological University. You help B.Tech students understand "
            "algorithms, Python programming, and problem-solving concepts.\n\n"
            
            "CORE PRINCIPLES:\n"
            "1. **Accuracy First**: Base all answers on the provided textbook context\n"
            "2. **Educational Focus**: Explain concepts clearly with step-by-step reasoning\n"
            "3. **Code Quality**: Provide well-commented, educational Python code\n"
            "4. **Practical Examples**: Use real-world examples when possible\n\n"
            
            "RESPONSE GUIDELINES:\n"
            "- If information is in the textbook context, provide detailed explanations\n"
            "- For code requests, use textbook examples first, then create educational examples\n"
            "- Break down complex concepts into digestible parts\n"
            "- Include both theory and practical applications\n"
            "- If information is not in context, say 'This topic may require additional reference from the textbook'\n"
        )
        
        if has_conversation_history:
            base_message += (
                "\nCONVERSATION AWARENESS:\n"
                "- Reference previous discussion when relevant\n"
                "- Build upon earlier explanations\n"
                "- Clarify or expand on previous answers when asked\n"
                "- Maintain context continuity throughout the conversation\n"
            )
        
        return base_message
    
    def generate_response(self, query: str, messages: List[Dict]) -> Any:
        """Generate enhanced response with conversation awareness."""
        try:
            # Get enhanced context
            textbook_context, conversation_context = self.get_enhanced_context(query, messages)
            
            # Prepare messages for API
            api_messages = []
            
            # System message
            has_history = len(messages) > 0
            system_msg = self.create_enhanced_system_message(has_history)
            api_messages.append({"role": "system", "content": system_msg})
            
            # Add conversation history (limited)
            recent_messages = messages[-self.config.max_chat_history:] if messages else []
            for msg in recent_messages:
                api_messages.append({
                    "role": msg["role"], 
                    "content": msg["content"][:800]  # Truncate long messages
                })
            
            # Create comprehensive prompt
            prompt_parts = []
            
            if textbook_context:
                prompt_parts.append(f"**Textbook Context:**\n{textbook_context}")
            
            if conversation_context and has_history:
                prompt_parts.append(f"**Previous Conversation:**\n{conversation_context}")
            
            prompt_parts.append(f"**Current Question:** {query}")
            
            if self.is_vague_query(query) and has_history:
                prompt_parts.append("*Note: This seems to be a follow-up question. Please provide context from our previous discussion.*")
            
            full_prompt = "\n\n".join(prompt_parts)
            api_messages.append({"role": "user", "content": full_prompt})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=api_messages,
                stream=True,
                max_tokens=1500,
                temperature=0.3,  # Lower temperature for more consistent responses
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

class StreamlitUI:
    """Enhanced Streamlit user interface."""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_custom_css()
        self.setup_sidebar()
    
    def setup_page_config(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="ATP Bot - Algorithm Thinking with Python",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_custom_css(self):
        """Enhanced CSS styling."""
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem !important;
                color: #1f77b4 !important;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .chat-message {
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 4px solid #1f77b4;
                background-color: #f8f9fa;
            }
            
            .user-message {
                background-color: #e3f2fd;
                border-left-color: #2196f3;
            }
            
            .assistant-message {
                background-color: #f1f8e9;
                border-left-color: #4caf50;
            }
            
            .context-indicator {
                font-size: 0.8rem;
                color: #666;
                font-style: italic;
                margin-bottom: 0.5rem;
            }
            
            .stTextInput > div > div > input {
                border-radius: 25px;
                border: 2px solid #e0e0e0;
                padding: 0.75rem 1rem;
                font-size: 1rem;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #1f77b4;
                box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
            }
            
            .stButton > button {
                border-radius: 10px;
                border: none;
                background: grey;
                color: white;
                font-weight: bold;
                padding: 0.5rem 1rem;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: scale(1.1);
                color: black;
                background-color: white;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            .status-success {
                color: #4caf50;
                font-weight: bold;
            }
            
            .status-error {
                color: #f44336;
                font-weight: bold;
            }
            
            .conversation-stats {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_sidebar(self):
        """Enhanced sidebar with conversation statistics."""
        st.sidebar.title("📚 Algorithm Thinking with Python")
        st.sidebar.markdown("""
        **Course Objectives:**
        - Develop algorithmic thinking skills
        - Master Python programming fundamentals  
        - Apply algorithms to real-world problems
        - Build problem-solving confidence
        """)
        
        st.sidebar.title("Smart Features")
        st.sidebar.markdown("""
        **Conversation Memory:**
        - Remembers previous questions
        - Provides contextual follow-ups
        - Builds on earlier explanations
        
        **Enhanced Search:**
        - Finds most relevant content
        - Prioritizes code examples
        - Adapts to your learning style
        """)
        
        # Conversation statistics
        if "messages" in st.session_state and st.session_state.messages:
            st.sidebar.markdown("### Session Stats")
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            
            st.sidebar.metric("Questions Asked", user_messages)
            st.sidebar.metric("Total Exchanges", total_messages // 2)
            
            # Show recent topics
            if user_messages > 0:
                recent_query = next((m["content"][:50] + "..." for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
                st.sidebar.markdown(f"**Recent:** {recent_query}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("💡 **Tips for Better Results:**")
        st.sidebar.markdown("""
        ✅ **Ask specific questions** about topics  
        ✅ **Request code examples** when learning  
        ✅ **Use follow-up questions** for clarification  
        ✅ **Reference previous answers** for deeper understanding  
        """)

class ATPBotApp:
    """Enhanced main application class."""
    
    def __init__(self):
        self.config = ATPBotConfig()
        self.ui = StreamlitUI()
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_store = EnhancedVectorStore(self.config)
        self.chatbot = EnhancedChatBot(self.config, self.vector_store)
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_context" not in st.session_state:
            st.session_state.last_context = ""
        if "session_start_time" not in st.session_state:
            st.session_state.session_start_time = datetime.now()
    
    def check_and_process_document(self) -> bool:
        """Enhanced document processing with progress tracking."""
        try:
            with st.spinner("🔄 Checking textbook status..."):
                current_hash = self.doc_processor.get_pdf_hash(self.config.pdf_path)
                previous_hash = self.doc_processor.load_hash()
                
                if current_hash != previous_hash:
                    st.info("Processing textbook content...")
                    
                    # Progress bar for processing
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load and process chunks
                    status_text.text("Loading textbook content...")
                    progress_bar.progress(25)
                    
                    chunks, metadata = self.doc_processor.load_pdf_chunks(self.config.pdf_path)
                    
                    if not chunks:
                        st.error("❌ No content found in textbook file")
                        return False
                    
                    status_text.text("Creating vector embeddings...")
                    progress_bar.progress(75)
                    
                    # Store in vector database
                    self.vector_store.store_chunks(chunks, metadata)
                    self.doc_processor.save_hash(current_hash)
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    st.success(f"✅ Textbook processed! {len(chunks)} chunks indexed with enhanced search capabilities.")
                else:
                    st.success("✅ Textbook is up-to-date and ready for enhanced conversations!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            st.error("❌ Error processing textbook. Please check the file and try again.")
            return False
    
    def display_chat_history(self):
        """Enhanced chat history display with context indicators."""
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                # Add context indicator for assistant responses
                if message["role"] == "assistant" and i > 0:
                    prev_msg = st.session_state.messages[i-1]
                    if len(prev_msg["content"]) < 20:  # Likely a follow-up question
                        st.markdown('<div class="context-indicator">💭 Building on previous discussion</div>', 
                                  unsafe_allow_html=True)
                
                st.markdown(message["content"])
    
    def handle_user_input(self):
        """Enhanced user input handling with conversation awareness."""
        # Enhanced input placeholder
        placeholder_text = "What do you want to know?"
        if st.session_state.messages:
            placeholder_text = "Continue our discussion or ask a new question..."
        
        if user_query := st.chat_input(placeholder_text):
            # Validate and clean input
            user_query = user_query.strip()
            if not user_query:
                st.warning("Please enter a valid question.")
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Generate and display response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Show thinking indicator
                    thinking_messages = [
                        "Analyzing your question...",
                        "Searching textbook content...",
                        "Preparing response..."
                    ]
                    
                    with st.spinner(thinking_messages[len(st.session_state.messages) % len(thinking_messages)]):
                        response_stream = self.chatbot.generate_response(user_query, st.session_state.messages[:-1])
                    
                    # Stream response with enhanced formatting
                    for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            
                            # Enhanced streaming with better formatting
                            formatted_response = self._format_response_content(full_response)
                            response_placeholder.markdown(formatted_response + "▌")
                    
                    # Final formatting
                    final_response = self._format_response_content(full_response)
                    response_placeholder.markdown(final_response)
                    
                    # Add to session state
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Update conversation statistics
                    self._update_conversation_stats()
                
                except Exception as e:
                    logger.error(f"Error handling user input: {str(e)}")
                    error_message = "⚠️ I encountered an error while processing your question. Please try rephrasing or ask a different question."
                    response_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    def _format_response_content(self, content: str) -> str:
        """Format response content for better display."""
        # Add syntax highlighting hints for code blocks
        formatted_content = content
        
        # Ensure code blocks are properly formatted
        if "```python" not in formatted_content and ("def " in formatted_content or "import " in formatted_content):
            # Wrap code-like content in proper markdown
            lines = formatted_content.split('\n')
            in_code_block = False
            formatted_lines = []
            
            for line in lines:
                if any(keyword in line for keyword in ["def ", "class ", "import ", "for ", "if ", "while "]):
                    if not in_code_block:
                        formatted_lines.append("```python")
                        in_code_block = True
                    formatted_lines.append(line)
                elif in_code_block and line.strip() == "":
                    formatted_lines.append(line)
                elif in_code_block and not line.startswith(" ") and not line.startswith("\t"):
                    formatted_lines.append("```")
                    formatted_lines.append(line)
                    in_code_block = False
                else:
                    formatted_lines.append(line)
            
            if in_code_block:
                formatted_lines.append("```")
            
            formatted_content = '\n'.join(formatted_lines)
        
        return formatted_content
    
    def _update_conversation_stats(self):
        """Update conversation statistics."""
        if "conversation_stats" not in st.session_state:
            st.session_state.conversation_stats = {
                "total_questions": 0,
                "code_examples_shown": 0,
                "topics_covered": set()
            }
        
        # Update stats based on latest interaction
        if st.session_state.messages:
            latest_user_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
            latest_assistant_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
            
            if latest_user_msg:
                st.session_state.conversation_stats["total_questions"] += 1
                
                # Extract topics (simple keyword extraction)
                content = latest_user_msg["content"].lower()
                topics = re.findall(r'\b(algorithm|function|loop|variable|list|dict|class|object|string|integer|float|boolean|conditional|iteration|recursion|sorting|searching)\b', content)
                st.session_state.conversation_stats["topics_covered"].update(topics)
            
            if latest_assistant_msg and "```python" in latest_assistant_msg["content"]:
                st.session_state.conversation_stats["code_examples_shown"] += 1
    
    def display_conversation_insights(self):
        """Display conversation insights and suggestions."""
        if len(st.session_state.messages) >= 4:  # After at least 2 exchanges
            stats = st.session_state.conversation_stats
            
            with st.expander("Conversation Insights", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Questions Asked", stats.get("total_questions", 0))
                
                with col2:
                    st.metric("Code Examples", stats.get("code_examples_shown", 0))
                
                with col3:
                    st.metric("Topics Covered", len(stats.get("topics_covered", set())))
                
                # Topic suggestions
                if stats.get("topics_covered"):
                    st.markdown("**Topics Discussed:** " + ", ".join(list(stats["topics_covered"])[:5]))
                
                # Learning suggestions
                st.markdown("**💡 Suggested Next Steps:**")
                suggestions = self._generate_learning_suggestions(stats)
                for suggestion in suggestions:
                    st.markdown(f"• {suggestion}")
    
    def _generate_learning_suggestions(self, stats: dict) -> List[str]:
        """Generate personalized learning suggestions."""
        suggestions = []
        topics_covered = stats.get("topics_covered", set())
        code_examples = stats.get("code_examples_shown", 0)
        
        # Suggest related topics
        if "algorithm" in topics_covered:
            suggestions.append("Try asking about specific algorithms like bubble sort or binary search")
        
        if "function" in topics_covered and "recursion" not in topics_covered:
            suggestions.append("Explore recursion to deepen your understanding of functions")
        
        if "loop" in topics_covered and "algorithm" not in topics_covered:
            suggestions.append("Learn how loops are used in different algorithms")
        
        if code_examples == 0:
            suggestions.append("Ask for code examples to see concepts in action")
        
        if len(topics_covered) >= 3:
            suggestions.append("Try asking for a comprehensive example that combines multiple concepts")
        
        # Default suggestions
        if not suggestions:
            suggestions = [
                "Ask about Python data structures like lists and dictionaries",
                "Explore algorithm complexity and Big O notation",
                "Request code examples for better understanding"
            ]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def run(self):
        """Run the enhanced application."""
        # Header with enhanced styling
        st.markdown('''
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="main-header">Chat-Alpy</h1>
            <p style="font-size: 1.2rem; color: #666; margin-top: -1rem;">
                Your AI Teaching Assistant for Algorithm Thinking with Python
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Process document
        if not self.check_and_process_document():
            st.stop()
        
        # Main chat interface
        st.markdown("### 👨🏻‍💻 Enhanced Chat Experience")
        
        # Display conversation insights
        self.display_conversation_insights()
        
        # Chat history
        self.display_chat_history()
        
        # User input
        self.handle_user_input()
        
        # Enhanced controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.last_context = ""
                st.session_state.conversation_stats = {
                    "total_questions": 0,
                    "code_examples_shown": 0,
                    "topics_covered": set()
                }
                st.rerun()
        
        with col2:
            if st.button("New Session"):
                # Reset session but keep processed documents
                for key in list(st.session_state.keys()):
                    if key not in ["pdf_processed", "chunks_stored"]:
                        del st.session_state[key]
                st.rerun()
        
        # Quick action buttons
        if not st.session_state.messages:
            st.markdown("### 🚀 Quick Start")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Basic Python Concepts"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "Explain basic Python programming concepts for beginners"
                    })
                    st.rerun()
            
            with col2:
                if st.button("Algorithm Fundamentals"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "What are the fundamental concepts of algorithms I should know?"
                    })
                    st.rerun()
            
            with col3:
                if st.button("Code Examples"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "Show me some Python code examples for common algorithms"
                    })
                    st.rerun()

def main():
    """Main function with enhanced error handling."""
    try:
        # Initialize and run app
        app = ATPBotApp()
        app.run()
        
    except ValueError as ve:
        st.error("❌ Configuration Error: " + str(ve))
        st.info("💡 Please check your environment variables and try again.")
        
    except FileNotFoundError as fe:
        st.error("❌ File Error: " + str(fe))
        st.info("💡 Please ensure the ATP_Split.txt file is in the correct location.")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("❌ Unexpected error occurred. Please refresh the page and try again.")
        
        # Show detailed error in development
        if os.getenv("DEBUG", "false").lower() == "true":
            st.exception(e)

if __name__ == "__main__":
    main()
