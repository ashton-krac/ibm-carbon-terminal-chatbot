# Carbon Design System RAG Chatbot Project

## Overview
This project creates a chatbot that can answer questions about IBM's Carbon Design System using RAG (Retrieval Augmented Generation). The chatbot provides accurate answers by referencing the actual Carbon documentation rather than relying solely on the LLM's training data.

## Core Components

### 1. Data Source
- Uses `ibm_carbon_v1.json` which contains IBM Carbon Design System documentation
- Each document in the JSON has:
  - `title`: Name of the documentation page
  - `url`: Source URL
  - `content`: Actual documentation content

### 2. Vector Database Setup (`setup_carbon_db.py`)
Key components:
- Loads the Carbon JSON documentation
- Creates embeddings using OpenAI's embedding model
- Stores embeddings in a Chroma vector database
- Persists the database to disk for reuse

### 3. RAG Chatbot (`carbon_chatbot.py`)
The chatbot implements the RAG process:

#### Retrieval
```python
docs = self.vectorstore.similarity_search(question, k=2)
```
- Uses Chroma to find relevant documentation
- Converts user question into embeddings
- Finds most similar documentation chunks

#### Augmentation
```python
context = "\n".join(
    f"From {doc.metadata['title']}:\n{doc.page_content}\n"
    for doc in docs
)
```
- Takes retrieved documentation
- Formats it into context
- Adds it to the prompt

#### Generation
```python
self.llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    streaming=True
)
```
- Uses GPT-4o to generate answers
- Streams responses for better UX
- Bases answers on retrieved documentation

## Key Features
1. **Accuracy**: 
   - Answers based on actual Carbon documentation
   - Admits when information isn't found

2. **User Experience**:
   - Interactive command-line interface
   - Streaming responses
   - Simple question/answer format

3. **Technical Features**:
   - Vector similarity search
   - Document embedding
   - Response streaming
   - Environment variable configuration

## Project Structure
```
project/
├── .env                    # OpenAI API key
├── ibm_carbon_v1.json     # Carbon documentation
├── setup_carbon_db.py     # Database setup script
├── carbon_chatbot.py      # Main chatbot script
└── vector_db/            # Generated vector database
```

## Required Packages
```bash
langchain
langchain-openai
langchain-chroma
python-dotenv
```

## Setup Instructions

1. Environment Setup:
```bash
# Create .env file
OPENAI_API_KEY=your-api-key-here
```

2. Install Dependencies:
```bash
pip install langchain langchain-openai langchain-chroma python-dotenv
```

3. Create Vector Database:
```bash
python setup_carbon_db.py
```

4. Run Chatbot:
```bash
python carbon_chatbot.py
```

## Usage Example
```
Carbon Design System ChatBot (type 'exit' to quit)
----------------------------------------

Your question: What is the hex color value for primary buttons?
Answer: [Streams response based on Carbon documentation]

Your question: exit
Goodbye!
```

## How RAG Works in This Project

1. **When Setting Up**:
   - Documentation is converted to embeddings
   - Embeddings are stored in Chroma database
   - Each document maintains metadata (title, URL)

2. **When Asking Questions**:
   - Question is converted to embedding
   - Similar documentation is retrieved
   - Context is added to prompt
   - GPT-4 generates specific answer

3. **Benefits of RAG**:
   - More accurate than pure LLM responses
   - Grounded in actual documentation
   - Reduces hallucination
   - Provides source attribution

## Limitations and Considerations
1. Requires OpenAI API key
2. Limited to information in Carbon documentation
3. Vector database needs updating when documentation changes
4. Response quality depends on retrieval quality
