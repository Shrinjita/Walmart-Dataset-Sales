import os
import pandas as pd
import concurrent.futures
from functools import lru_cache
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
import streamlit as st
import time
import random
import textwrap

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Walmart Sales Analysis",
    page_icon="📊",
    layout="wide"
)

# Constants
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
MODEL_VARIANTS = ["gemini-1.5-pro-latest"]
CHUNK_SIZE = 100
BATCH_SIZE = 50

# Load Walmart data with caching and progress
@st.cache_data
def load_data():
    try:
        with st.spinner("Loading Walmart data..."):
            data = pd.read_csv("Walmart.csv")
            st.session_state.data_loaded = True
            return data
    except Exception as e:
        st.error(f"Failed to load Walmart.csv: {str(e)}")
        st.stop()

walmart_data = load_data()

# Configure Gemini with load balancing
@st.cache_resource
def configure_gemini():
    GEMINI_API_KEY = os.getenv("API_KEY")
    if not GEMINI_API_KEY:
        st.error("API_KEY not found in environment variables")
        st.stop()
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    return {
        "EMBEDDING_MODEL": "models/embedding-001",
        "MODEL_VARIANTS": MODEL_VARIANTS,
        "last_used_model": 0
    }

config = configure_gemini()

# Predefined questions organized by category
QUESTIONS = {
    "Overall Performance": [
        "How does weekly sales performance vary across different stores?",
        "What are the key drivers of sales growth or decline?",
        "How do holiday weeks compare to non-holiday weeks in terms of sales?"
    ],
    "Product Analysis": [
        "What is the distribution of sales across different departments?",
        "Which departments show the strongest seasonal patterns?",
        "How does temperature affect sales of different departments?"
    ],
    "Customer Insights": [
        "What are the characteristics of stores with highest sales?",
        "How does fuel price correlate with customer spending?",
        "What external factors most strongly influence sales?"
    ]
}

# Initialize vector store with optimized chunking
@st.cache_resource
def init_vector_db():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("walmart_sales")
        
        if collection.count() == 0:
            # Create optimized chunks
            data_chunks = []
            for i in range(0, len(walmart_data), CHUNK_SIZE):
                chunk = walmart_data.iloc[i:i+CHUNK_SIZE].to_csv(index=False)
                data_chunks.append(chunk)
            
            # Parallel embedding generation with retries
            def generate_embedding(chunk):
                for attempt in range(MAX_RETRIES):
                    try:
                        response = genai.embed_content(
                            model=config["EMBEDDING_MODEL"],
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        return response['embedding']
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            raise e
                        time.sleep((BACKOFF_FACTOR ** attempt) + random.uniform(0, 1))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                embeddings = list(executor.map(generate_embedding, data_chunks))
            
            # Batch add to collection
            for i in range(0, len(data_chunks), BATCH_SIZE):
                collection.add(
                    embeddings=embeddings[i:i+BATCH_SIZE],
                    documents=data_chunks[i:i+BATCH_SIZE],
                    ids=[f"doc_{j}" for j in range(i, min(i+BATCH_SIZE, len(data_chunks)))]
                )
        
        return collection
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

# Enhanced response generation with formatting
def generate_response_with_retry(prompt: str) -> str:
    last_exception = None
    
    # Round-robin model selection
    config["last_used_model"] = (config["last_used_model"] + 1) % len(config["MODEL_VARIANTS"])
    current_model = config["MODEL_VARIANTS"][config["last_used_model"]]
    
    for attempt in range(MAX_RETRIES):
        try:
            model = genai.GenerativeModel(current_model)
            response = model.generate_content(
                f"""Format your response with proper paragraphs and numerical formatting.
                Avoid mid-word or mid-number line breaks.
                {prompt}""",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=500,
                    response_mime_type="text/plain"
                )
            )
            return format_response(response.text)
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                time.sleep((BACKOFF_FACTOR ** attempt) + random.uniform(0, 1))
    
    raise last_exception if last_exception else Exception("Failed to generate response")

@lru_cache(maxsize=100)
def generate_response(prompt: str) -> str:
    return generate_response_with_retry(prompt)

def format_response(text: str) -> str:
    """Clean and format the response text"""
    # Fix common formatting issues
    text = ' '.join(text.split())  # Remove extra whitespace
    text = text.replace(' ,', ',').replace(' .', '.')  # Fix punctuation spacing
    
    # Ensure numbers stay together
    for match in set(text.split()):
        if match.isdigit() and len(match) > 3:
            text = text.replace(match, match.replace(' ', ''))
    
    # Add proper paragraph breaks
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    paragraphs = []
    current_para = []
    
    for i, sentence in enumerate(sentences):
        current_para.append(sentence)
        if (i + 1) % 3 == 0 or i == len(sentences) - 1:
            paragraphs.append('. '.join(current_para) + '.')
            current_para = []
    
    return '\n\n'.join(paragraphs)

def display_data_overview():
    with st.expander("🔍 Data Overview", expanded=False):
        st.dataframe(walmart_data.head(10))
        st.write(f"**Total Records:** {len(walmart_data)}")
        st.write(f"**Date Range:** {walmart_data['Date'].min()} to {walmart_data['Date'].max()}")

def predefined_analysis(collection):
    st.subheader("📋 Predefined Analysis")
    category = st.selectbox("Select category:", list(QUESTIONS.keys()))
    question = st.selectbox("Select question:", QUESTIONS[category])
    
    if st.button("🚀 Generate Analysis"):
        with st.spinner("Analyzing..."):
            try:
                start_time = time.time()
                prompt = f"""Analyze this Walmart sales data:
                {walmart_data.describe().to_string()}
                
                Provide a well-formatted response to:
                {question}
                
                Include:
                - Key findings with specific numbers
                - Clear paragraphs
                - Actionable recommendations"""
                
                response = generate_response(prompt)
                
                st.subheader("📈 Analysis Results")
                st.markdown(response)
                st.caption(f"Generated in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

def interactive_chat(collection):
    st.subheader("🤖 Interactive Analysis")
    st.caption("Ask any question about the Walmart sales data")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle new query
    if user_query := st.chat_input("Your question..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.spinner("Analyzing..."):
            try:
                start_time = time.time()
                
                # Get context
                query_embedding = genai.embed_content(
                    model=config["EMBEDDING_MODEL"],
                    content=user_query,
                    task_type="retrieval_query"
                )['embedding']
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2
                )
                context = "\n".join(results['documents'][0])
                
                # Generate response
                prompt = f"""Context: {context}
                
                Question: {user_query}
                
                Provide a concise, well-formatted answer with:
                - Proper paragraph breaks
                - Intact numbers and statistics
                - Clear bullet points"""
                
                response = generate_response(prompt)
                
                # Store and display
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "time": time.time() - start_time
                })
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.caption(f"Generated in {st.session_state.chat_history[-1]['time']:.2f}s")
                    
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)

def main():
    st.title("📊 Walmart Sales Analytics")
    display_data_overview()
    
    collection = init_vector_db()
    if not collection:
        st.error("Failed to initialize data storage")
        return
    
    predefined_analysis(collection)
    st.divider()
    interactive_chat(collection)

if __name__ == "__main__":
    main()