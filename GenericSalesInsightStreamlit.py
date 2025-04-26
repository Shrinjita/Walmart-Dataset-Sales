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

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Walmart Sales Analysis",
    page_icon="📊",
    layout="wide"
)

# Load balancing configuration
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
MODEL_VARIANTS = ["gemini-1.5-pro-latest", "gemini-1.0-pro-latest"]  # Fallback models

# Load Walmart data with caching
@st.cache_data
def load_data():
    try:
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
        "last_used_model": 0  # For round-robin selection
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

# Initialize vector store with Walmart data using parallel processing
@st.cache_resource
def init_vector_db():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("walmart_sales")
        
        if collection.count() == 0:  # Only populate if empty
            # Create chunks from the data
            data_chunks = []
            chunk_size = min(100, len(walmart_data)//10  # Dynamic chunk sizing
            for i in range(0, len(walmart_data), chunk_size):
                chunk = walmart_data.iloc[i:i+chunk_size].to_csv(index=False)
                data_chunks.append(chunk)
            
            # Parallel embedding generation
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
                        time.sleep(BACKOFF_FACTOR ** attempt)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                embeddings = list(executor.map(generate_embedding, data_chunks))
            
            # Batch add to collection
            batch_size = 100
            for i in range(0, len(data_chunks), batch_size):
                collection.add(
                    embeddings=embeddings[i:i+batch_size],
                    documents=data_chunks[i:i+batch_size],
                    ids=[f"doc_{j}" for j in range(i, min(i+batch_size, len(data_chunks)))]
                )
        
        return collection
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

# Load-balanced response generation
def generate_response_with_retry(prompt: str) -> str:
    last_exception = None
    
    # Round-robin model selection
    config["last_used_model"] = (config["last_used_model"] + 1) % len(config["MODEL_VARIANTS"])
    current_model = config["MODEL_VARIANTS"][config["last_used_model"]]
    
    for attempt in range(MAX_RETRIES):
        try:
            model = genai.GenerativeModel(current_model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=350
                )
            )
            return response.text
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                # Exponential backoff
                sleep_time = (BACKOFF_FACTOR ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
    
    raise last_exception if last_exception else Exception("Unknown error")

@lru_cache(maxsize=100)
def generate_response(prompt: str) -> str:
    return generate_response_with_retry(prompt)

def main_content(collection):
    st.title("📊 Walmart Sales Analysis")
    
    # Show data summary
    with st.expander("🔍 View Walmart Data Sample"):
        st.dataframe(walmart_data.head())
        st.caption(f"Total records: {len(walmart_data)}")
    
    # Predefined questions section
    st.subheader("📋 Predefined Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox(
            "Select analysis category:",
            list(QUESTIONS.keys()),
            index=0
        )
    
    with col2:
        question = st.selectbox(
            "Select your question:",
            QUESTIONS[category],
            index=0
        )
    
    if st.button("🚀 Generate Analysis", key="predefined_btn"):
        with st.spinner("Analyzing Walmart data..."):
            try:
                prompt = f"""Using this Walmart sales data:
                {walmart_data.describe().to_string()}
                
                Analyze and provide:
                - Concise 100-word summary
                - 3 key insights with specific numbers
                - 2 actionable recommendations
                
                Question: {question}"""
                
                response = generate_response(prompt)
                
                st.subheader("📈 Analysis Results")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Chatbot section
    st.divider()
    st.subheader("🤖 Interactive Analysis")
    st.caption("Ask any question about the Walmart sales data")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_query := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate response
        with st.spinner("Analyzing your question..."):
            try:
                # Get relevant context in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    embedding_future = executor.submit(
                        genai.embed_content,
                        model=config["EMBEDDING_MODEL"],
                        content=user_query,
                        task_type="retrieval_query"
                    )
                    
                    # Prepare prompt while waiting for embedding
                    prompt = f"""Provide a concise 150-word answer to:
                    {user_query}
                    
                    Include:
                    - Specific numbers from the data when possible
                    - Actionable insights
                    - Relevant trends"""
                    
                    # Get embedding result
                    query_embedding = embedding_future.result()['embedding']
                    
                    # Query collection
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=2
                    )
                    context = "\n".join(results['documents'][0])
                    
                    # Finalize prompt with context
                    prompt = f"""Using this Walmart sales data:
                    {context}
                    
                    {prompt}"""
                
                # Generate response with load balancing
                response = generate_response(prompt)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)

# Main app
def main():
    collection = init_vector_db()
    if collection:
        main_content(collection)
    else:
        st.error("Failed to initialize data storage")

if __name__ == "__main__":
    main()