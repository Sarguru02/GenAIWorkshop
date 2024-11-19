# Question Answering System Using Dense Retrieval and BERT

## Overview
This project implements a question answering (QA) system that combines dense document retrieval with a BERT model for generating answers. The system retrieves relevant context from a knowledge base using FAISS and provides answers using a pre-trained BERT model fine-tuned for question answering tasks.

## Features
- **Dense Retrieval**: Uses `SentenceTransformer` to create embeddings for documents and FAISS for efficient similarity search.
- **BERT for QA**: A pre-trained BERT model is used to extract answers from the retrieved context.
- **End-to-End Pipeline**: The system processes a user query, retrieves the most relevant document, and generates an answer.

## Code Components

### 1. Knowledge Base
A sample knowledge base is initialized as a list of text snippets:
```python
knowledge_base = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris, France.",
    "Python is a programming language that is widely used for web development, machine learning, and data science.",
    "Albert Einstein developed the theory of relativity.",
    "The sun rises in the east and sets in the west."
]
```

### 2. Embedding Generation
Embeddings for the knowledge base are generated using `SentenceTransformer`:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(knowledge_base)
```

### 3. FAISS Index Creation
A FAISS index is created to store document embeddings for fast retrieval:
```python
dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(document_embeddings)
```

### 4. Question Answering Model
A BERT model pre-trained on SQuAD is loaded for question answering:
```python
model_qa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
```

### 5. Functions
- **`retrieve_top_document(query, faiss_index, knowledge_base)`**: Retrieves the most relevant document based on the query.
- **`answer_question(context, question)`**: Generates an answer using BERT based on the retrieved context.

### 6. Main Execution
The script prompts the user for a question, retrieves the most relevant document, and generates an answer:
```python
if __name__ == "__main__":
    query = input("Enter your question: ")

    # Retrieve context
    context, score = retrieve_top_document(query, faiss_index, knowledge_base)
    print(f"Retrieved Context: {context} (Score: {score})")

    # Generate answer
    answer = answer_question(context, query)
    print(f"Answer: {answer}")
```

## Requirements
Install the required libraries with:
```bash
pip install faiss-cpu sentence-transformers transformers torch
```

## Usage
1. Clone this repository.
2. Run the script using Python:
   ```bash
   python qa_system.py
   ```
3. Enter a question when prompted and receive an answer based on the knowledge base.

## How It Works
1. **Embedding Creation**: The `SentenceTransformer` model creates embeddings for all documents in the knowledge base.
2. **Retrieval**: FAISS performs a similarity search to retrieve the most relevant document.
3. **Answer Generation**: The pre-trained BERT model processes the question and context to generate an answer.


