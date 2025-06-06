# Walmart Sales Analysis Project

This repository contains a suite of tools for analyzing Walmart sales data using Google's Gemini AI models. The project includes data preprocessing utilities, chatbot interfaces, and interactive visualizations to gain insights from Walmart's retail data.

## 📊 Project Overview

This project provides multiple tools for analyzing Walmart sales data:
- Interactive Streamlit dashboard for visualizations and AI-powered insights
- Command-line chatbot for quick sales analysis
- Data preprocessing utilities for embedding generation
- Vector database for efficient data retrieval

## 📋 Requirements

### Dataset
You need to download the Walmart dataset from Kaggle:
1. Visit [Walmart Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
2. Download and place `Walmart.csv` in the project root directory

### Installation
1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Google API key:

```
API_KEY=your_gemini_api_key_here
```

## 📁 File Structure

```
.
├── GeminiSalesInsightsChatbot.py    # Command-line chatbot for sales analysis
├── GenericSalesInsightStreamlit.py  # Generic Streamlit dashboard template
├── Walmart.csv                      # Dataset file (download from Kaggle)
├── basicGeminiSalesChatbot.py       # Simplified Gemini chatbot version
├── data_preprocessor.py             # Data preprocessing utility
├── preprocessing.log                # Log file for preprocessing operations
├── requirements.txt                 # Python dependencies
├── stepproj.py                      # Additional visualizations
├── walmart_analysis_app.py          # Main Streamlit dashboard application
├── walmart_insights.txt             # Example insights generated by the system
├── chroma_db/                       # Vector database directory (created during preprocessing)
├── data_chunks.json                 # Chunked data (created during preprocessing)
├── data_embeddings.json             # Data embeddings (created during preprocessing)
```

## 🚀 Getting Started

### Step 1: Preprocess the Data
Run the data preprocessor to generate embeddings and set up the vector database:

```bash
python data_preprocessor.py
```

Note: This will create the necessary `chroma_db` directory, `data_chunks.json`, and `data_embeddings.json` files.

### Step 2: Run the Main Application
Launch the interactive Streamlit dashboard:

```bash
streamlit run walmart_analysis_app.py
```

## 📝 Component Descriptions

### Main Applications

#### `walmart_analysis_app.py`
The primary Streamlit dashboard with:
- Interactive chat interface for custom queries
- Predefined analysis questions
- Data overview and visualization options
- RAG (Retrieval-Augmented Generation) system for accurate responses

#### `GeminiSalesInsightsChatbot.py`
A command-line chatbot that:
- Answers predefined sales analysis questions
- Provides concise, data-driven insights
- Saves analysis to text files

#### `stepproj.py`
Complementary visualization dashboard with:
- Category contribution analysis (pie charts)
- Sales trend analysis (line charts)
- Performance benchmarking (bar charts)
- Customer purchase distribution (histograms)

### Utility Files

#### `data_preprocessor.py`
Handles data preparation:
- Chunks Walmart data for efficient processing
- Generates embeddings using Gemini's embedding model
- Stores data in ChromaDB for vector search

#### `basicGeminiSalesChatbot.py`
A simplified version of the chatbot that:
- Demonstrates core RAG functionality
- Provides basic business insights
- Uses minimal dependencies

#### `GenericSalesInsightStreamlit.py`
A template version with:
- Load balancing across multiple Gemini models
- Parallel processing for efficiency
- Structured analysis categories

## 📈 Features

- **AI-Powered Insights**: Leverage Google's Gemini models for in-depth sales analysis
- **Interactive Queries**: Ask custom questions about your sales data
- **Predefined Analysis**: Quick access to common business questions
- **Visualizations**: Multiple chart types for data visualization
- **Resilient Design**: Load balancing, retries, and error handling

## 🔧 Troubleshooting

If you encounter the date format error seen in `preprocessing.log`, modify the date parsing in `data_preprocessor.py` to handle the "DD-MM-YYYY" format.

## 📚 Future Improvements

- Add additional visualization types
- Support for multiple data sources
- Export functionality for reports
- User authentication for team usage
- Batch processing for large datasets

## 📄 Demo
![image](https://github.com/user-attachments/assets/fd7f147b-1163-47c3-8c38-7e0025506198)
