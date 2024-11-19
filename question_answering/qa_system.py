import faiss
from sentence_transformers import SentenceTransformer
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Sample knowledge base
knowledge_base = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris, France.",
    "Python is a programming language that is widely used for web development, machine learning, and data science.",
    "Albert Einstein developed the theory of relativity.",
    "The sun rises in the east and sets in the west."
]

# Step 1: Create a SentenceTransformer model for generating document embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Generate embeddings for the knowledge base
document_embeddings = model.encode(knowledge_base)

# Step 3: Create a FAISS index
dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(document_embeddings)

# Load pre-trained BERT model and tokenizer for Question Answering
model_qa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Function to retrieve the most relevant document
def retrieve_top_document(query, faiss_index, knowledge_base):
    query_embedding = model.encode([query])  # Encode the query
    distances, indices = faiss_index.search(query_embedding, 1)  # Retrieve top document
    return knowledge_base[indices[0][0]], distances[0][0]

# Function for answering questions based on retrieved context
def answer_question(context, question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model_qa(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer_tokens = inputs.input_ids[0][start_index:end_index+1]
    answer = tokenizer.decode(answer_tokens)
    return answer

# Main execution
if __name__ == "__main__":
    query = input("Enter your question: ")

    # Retrieve the most relevant document from the knowledge base
    context, score = retrieve_top_document(query, faiss_index, knowledge_base)
    print(f"Retrieved Context: {context} (Score: {score})")

    # Generate an answer using BERT
    answer = answer_question(context, query)
    print(f"Answer: {answer}")
