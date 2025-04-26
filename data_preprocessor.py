# data_preprocessor.py
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import chromadb

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

def preprocess_and_chunk_data():
    # Load Walmart data
    data = pd.read_csv("Walmart.csv")
    
    # Create chunks
    chunk_size = 100
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size].to_dict(orient='records')
        chunks.append(chunk)
    
    # Generate embeddings using the correct method
    embeddings = []
    for chunk in chunks:
        # Updated embedding generation method
        response = genai.embed_content(
            model="models/embedding-001",
            content=str(chunk),
            task_type="retrieval_document"
        )
        embeddings.append(response['embedding'])
    
    # Save to files
    with open("data_chunks.json", "w") as f:
        json.dump(chunks, f)
    
    with open("data_embeddings.json", "w") as f:
        json.dump(embeddings, f)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("walmart_sales")
    
    # Add to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=[str(chunk) for chunk in chunks],
        ids=[f"doc_{i}" for i in range(len(chunks))]
    )
    
    print(f"Data preprocessing complete! Created {len(chunks)} chunks.")

if __name__ == "__main__":
    preprocess_and_chunk_data()