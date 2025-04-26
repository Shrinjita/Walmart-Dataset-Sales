import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
import chromadb
from functools import lru_cache
import textwrap

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("API_KEY not found in environment variables")

# Configure Gemini with correct model names
genai.configure(api_key=GEMINI_API_KEY)
GENERATION_MODEL = "gemini-1.5-pro-latest"
EMBEDDING_MODEL = "models/embedding-001"

# Predefined questions
QUESTIONS = [
    "How does overall sales performance compare to previous periods and market trends? (100 words)",
    "What are the key drivers of sales growth or decline? (100 words)",
    "Are there any significant deviations from expected performance? (100 words)",
    "Which products or services are generating the most revenue and profit? (100 words)",
    "Are there any emerging trends in product popularity? (100 words)",
    "Are there any underperforming products or services that need attention? (100 words)",
    "What are the characteristics of our most profitable customers? (100 words)",
    "Are there any underserved customer segments? (100 words)",
    "How can we better target different customer segments? (100 words)",
    "Who are the top-performing sales representatives? (100 words)",
    "What are the key factors contributing to their success? (100 words)",
    "Are there areas where sales reps need additional training? (100 words)",
    "What is the average sales cycle length? (100 words)",
    "How effective are our current sales strategies? (100 words)",
    "Are there any bottlenecks in our sales process? (100 words)",
    "How can we improve sales process efficiency? (100 words)",
    "How do we predict future sales trends from historical data? (100 words)",
    "What key factors influence our sales forecast? (100 words)",
    "How can we improve forecast accuracy? (100 words)",
    "What are the key drivers of customer churn? (100 words)",
    "How can we improve customer retention? (100 words)",
    "Which customers are most likely to churn? (100 words)"
]

def format_response(text, width=80):
    """Format response text for better readability"""
    return "\n".join(textwrap.wrap(text, width))

@lru_cache(maxsize=100)
def get_cached_response(prompt: str) -> str:
    """Cache responses using correct model name"""
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.8,
            max_output_tokens=350  # ~100 words
        )
    )
    return response.text

def answer_predefined_questions():
    """Answer all predefined questions sequentially"""
    answers = {}
    print("\n" + "="*80)
    print("WALMART SALES ANALYSIS REPORT".center(80))
    print("="*80 + "\n")
    
    for i, question in enumerate(QUESTIONS, 1):
        try:
            # Display progress
            print(f"\nQUESTION {i}/{len(QUESTIONS)}")
            print("-"*80)
            print(question.split("(100 words)")[0].strip())
            print("-"*80)
            
            # Generate answer
            prompt = f"""Analyze Walmart sales data and provide a concise, data-driven answer 
            in exactly 100 words. Focus on key insights and actionable recommendations:
            
            {question}"""
            
            answer = get_cached_response(prompt)
            answers[question] = answer
            
            # Display formatted answer
            print("\nANALYSIS:")
            print(format_response(answer))
            print(f"\n[Word count: {len(answer.split())}]")
            
            # Pause between questions except last one
            if i < len(QUESTIONS):
                input("\nPress Enter to continue to next question...")
                
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            answers[question] = f"Error: {str(e)}"
    
    return answers

def interactive_analysis(collection):
    """Interactive mode for user questions"""
    print("\n" + "="*80)
    print("INTERACTIVE ANALYSIS MODE".center(80))
    print("="*80)
    print("\nNow you can ask your own questions about the Walmart sales data.")
    print("Type 'exit' to end the session.\n")
    
    while True:
        try:
            user_question = input("\nYour question: ").strip()
            
            if user_question.lower() in ['exit', 'quit']:
                print("\nEnding interactive session...")
                break
                
            if len(user_question) < 10:
                print("Please ask a more detailed question (at least 10 characters)")
                continue
                
            # Generate answer with context
            model = genai.GenerativeModel(GENERATION_MODEL)
            response = model.generate_content(
                f"""Using Walmart sales data, provide a concise 100-150 word analysis:
                
                Question: {user_question}
                
                Focus on:
                1. Key data insights
                2. Actionable recommendations
                3. Supporting evidence from trends""",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    top_p=0.9,
                    max_output_tokens=500
                )
            )
            
            # Display formatted response
            print("\n" + "-"*80)
            print("ANALYSIS:".center(80))
            print("-"*80)
            print(format_response(response.text))
            print(f"\nResponse length: {len(response.text.split())} words")
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")

def main():
    try:
        # Initialize vector store (simplified for example)
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("sales_data")
        
        # Answer all predefined questions
        answers = answer_predefined_questions()
        
        # Save report to file
        with open("walmart_sales_analysis.txt", "w") as f:
            f.write("WALMART SALES ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            for q, a in answers.items():
                f.write(f"QUESTION: {q}\n")
                f.write("-"*80 + "\n")
                f.write(f"{a}\n\n")
                f.write("="*80 + "\n\n")
        
        # Start interactive session
        interactive_analysis(collection)
        
        print("\nAnalysis complete. Full report saved to 'walmart_sales_analysis.txt'")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()