import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import chromadb
from typing import List

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("API_KEY not found in environment variables")

# Configure Gemini models
genai.configure(api_key=GEMINI_API_KEY)
model_name = "gemini-1.5-flash"  # or "gemini-pro" for more complex analysis

def load_and_chunk_data(file_path: str, chunk_size: int = 100) -> List[str]:
    """Load data from CSV and chunk it into manageable pieces."""
    try:
        sales_data = pd.read_csv(file_path)
        return [
            sales_data[i:i + chunk_size].to_string() 
            for i in range(0, len(sales_data), chunk_size)
        ]
    except Exception as e:
        raise ValueError(f"Error loading or chunking data: {str(e)}")

def create_vector_store(data_chunks: List[str]) -> chromadb.Collection:
    """Create embeddings and store in ChromaDB vector database."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="sales_data",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Generate embeddings
        embeddings = []
        for chunk in data_chunks:
            response = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings.append(response['embedding'])
        
        collection.add(
            embeddings=embeddings,
            documents=data_chunks,
            ids=[f"doc_{i}" for i in range(len(data_chunks))]
        )
        return collection
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {str(e)}")

def generate_business_insight(query: str, collection: chromadb.Collection) -> str:
    """Generate business insights using RAG pattern."""
    try:
        # Get query embedding
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Retrieve relevant context
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        context = "\n".join(results['documents'][0])
        
        # Generate response
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            f"""Analyze the following Walmart sales data and generate a detailed business insight report of approximately 100 words.
            
            User Question: {query}
            
            Relevant Data:
            {context}
            
            Report should include:
            1. Key trends and patterns
            2. Notable anomalies or outliers
            3. Actionable recommendations
            4. Potential areas for further investigation""",
            generation_config=GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=1000
            )
        )
        
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error generating insights: {str(e)}")

def main():
    try:
        # 1. Load and prepare data
        data_chunks = load_and_chunk_data("Walmart.csv")
        
        # 2. Create vector store
        collection = create_vector_store(data_chunks)
        
        # 3. Example usage
        user_question = "What are the key trends in sales across different departments?"
        report = generate_business_insight(user_question, collection)
        print(report)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()