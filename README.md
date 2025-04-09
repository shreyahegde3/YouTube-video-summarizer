# Enhanced RAG & Agent Q&A System

An advanced question-answering system that combines Retrieval-Augmented Generation (RAG) and Agent-based reasoning with YouTube video integration for comprehensive educational responses.

![image](https://github.com/user-attachments/assets/b6cfbcc2-783b-4864-bd1a-3ddd2e06bea3)


## Overview

This system provides an interactive interface for answering educational queries by leveraging:

1. **Knowledge Base Retrieval**: Accesses embeddings stored in Pinecone to find relevant information
2. **YouTube Video Integration**: Extracts context from video transcripts for additional information
3. **Dynamic Knowledge Expansion**: Automatically expands the knowledge base by embedding relevant videos
4. **Advanced Reasoning Techniques**: Implements multiple prompting techniques for comprehensive answers

## Features

- **Dual Operational Modes**:
  - **RAG Mode**: Standard retrieval-augmented generation for direct, knowledge-based answers
  - **Agent Mode**: Advanced reasoning with deeper analysis and comprehensive explanations

- **Multiple Prompting Techniques**:
  - **Standard**: Basic reasoning for straightforward questions
  - **Chain of Thought (CoT)**: Step-by-step reasoning process
  - **Tree of Thought (ToT)**: Explores multiple solution paths
  - **Graph of Thought (GoT)**: Maps interconnected concepts and relationships

- **Dynamic Knowledge Base**:
  - Automatically expands by embedding YouTube video transcripts when relevant information is scarce
  - Measures relevance of existing knowledge and fetches supplementary information when needed

- **Smart Video Integration**:
  - Manual URL input option for specific videos
  - Automatic relevant video discovery based on query context
  - Transcript extraction and processing for contextual information

## How It Works

### RAG (Retrieval-Augmented Generation)
RAG combines traditional retrieval-based methods with generative AI. When a query is submitted, the system:
1. Converts the query into embeddings
2. Retrieves the most relevant information from the knowledge base
3. Uses this context to generate accurate, knowledge-grounded responses

### Agent Mode
The Agent mode enhances the RAG foundation with:
1. More sophisticated reasoning patterns
2. Multi-perspective analysis
3. Deep conceptual exploration
4. Structured problem decomposition
5. Comprehensive solution formulation

### Automatic Knowledge Expansion
When answering queries:
1. The system evaluates relevance scores from the knowledge base
2. If scores fall below threshold (0.4), it automatically processes relevant YouTube content
3. New embeddings are created and stored for future reference
4. Knowledge base continuously improves through use

## Technical Components

- **Backend**: Python with Gradio for the web interface
- **Embedding Model**: Sentence Transformers for semantic understanding
- **Vector Database**: Pinecone for efficient similarity search
- **LLM Integration**: Groq API with Llama 3 70B model
- **NLP Processing**: spaCy and NLTK for text analysis
- **Video Integration**: YouTube Transcript API

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```
   PINECONE_API_KEY=your_pinecone_key
   GROQ_API_KEY=your_groq_key
   YOUTUBE_API_KEY=your_youtube_key
   PINECONE_INDEX_NAME=embeddings
   ```
4. Run the application:
   ```bash
   python app_gradio.py
   ```

## Usage

1. Select between RAG or Agent mode
2. Choose a prompting technique
3. Enter your question
4. Optionally provide a YouTube video URL
5. Click "Get Answer"
6. Review both the answer and sources used


## Acknowledgments

- This system uses the Groq API for LLM access
- Pinecone for vector database services
- YouTube API for video information
