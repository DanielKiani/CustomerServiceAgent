import faiss
import numpy as np
import torch
import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class CustomerServiceAgent:
    """
    Encapsulates all the functionality of the AI customer service agent,
    including model loading, knowledge base preparation, and response generation.
    """
    def __init__(self):
        """
        Initializes the agent by loading all necessary models and building the
        retrieval-augmented generation (RAG) knowledge base.
        """
        print("Initializing Customer Service Agent...")
        self._load_models()
        self._build_knowledge_base()
        print("\nAgent is ready.")

    def _load_models(self):
        """
        Loads all the machine learning models required for the agent to function.
        """
        print("\n[1/4] Loading all models...")
        device = 0 if torch.cuda.is_available() else -1
        
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm_pipeline = pipeline("text2text-generation", model='google/flan-t5-large', device=device)
        self.sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
        
        print("All models loaded successfully.")

    def _build_knowledge_base(self):
        """
        Prepares the knowledge base for the RAG system by loading FAQs,
        creating vector embeddings, and storing them in a FAISS index.
        """
        print("\n[2/4] Preparing Knowledge Base...")
        try:
            dataset = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")
            self.knowledge_base = [item for item in dataset['answer'] if item and item.strip()]
            print(f"Successfully loaded {len(self.knowledge_base)} documents.")
        except Exception as e:
            print(f"Failed to load dataset. Using a fallback. Error: {e}")
            self.knowledge_base = [
                "You can update your payment method by going to the 'Billing' section in your account settings.",
                "To check your order status, please log in to your account and navigate to the 'My Orders' page.",
                "I am very sorry to hear your package has not arrived. Please provide your order number so I can investigate.",
            ]
        
        print("\n[3/4] Creating embeddings for the knowledge base...")
        embeddings = self.embedding_model.encode(self.knowledge_base, show_progress_bar=True)
        
        print("\n[4/4] Setting up FAISS vector index...")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        print("FAISS retriever is ready.")

    def get_rag_response(self, query, history, k=3):
        """
        Generates a response using the Retrieval-Augmented Generation (RAG) pipeline.
        """
        print(f"\nProcessing query: '{query}'")
        
        sentiment = self.sentiment_classifier(query)[0]['label']
        print(f"Detected Sentiment: {sentiment}")

        history_string = "".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}\n" for turn in history])

        query_embedding = self.embedding_model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k)
        context = "\n\n".join([self.knowledge_base[i] for i in indices[0]])

        persona = "You are an empathetic customer support agent." if sentiment == 'NEGATIVE' else "You are a helpful customer support agent."
        
        prompt = f"""
        {persona}
        Based on history and context, answer the user's question.

        ### History:
        {history_string}
        ### Context:
        {context}
        ### Question:
        {query}
        ### Answer:
        """
        
        start_time = time.time()
        response = self.llm_pipeline(prompt, max_new_tokens=100, num_beams=5, early_stopping=True)[0]['generated_text']
        print(f"LLM Response Time: {time.time() - start_time:.2f} seconds")
        
        return response.strip()

# --- Terminal-based Demo ---
if __name__ == "__main__":
    agent = CustomerServiceAgent()
    conversation_history = []
    
    print("\n--- Starting Terminal Demo ---")
    
    # First query
    query1 = "This is so frustrating, my package never arrived!"
    response1 = agent.get_rag_response(query1, conversation_history)
    conversation_history.append({'user': query1, 'assistant': response1})
    
    print(f"\nUser: {query1}")
    print(f"Agent: {response1}")
    
    # Follow-up query to test memory
    query2 = "Okay, what do you need from me to find it?"
    response2 = agent.get_rag_response(query2, conversation_history)
    conversation_history.append({'user': query2, 'assistant': response2})

    print(f"\nUser: {query2}")
    print(f"Agent: {response2}")
    
    print("\n--- Demo Complete ---")