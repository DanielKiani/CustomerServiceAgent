import faiss
import numpy as np
import torch
import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class CustomerServiceAgent:
    """
    AI Customer Service Agent with RAG + robust off-topic detection.
    """

    def __init__(self):
        print("Initializing IMPROVED Customer Service Agent...")
        self._load_models()
        self._build_knowledge_base()
        print("\nAgent is ready.")

    def _load_models(self):
        """
        Loads all the ML models required for the agent.
        """
        print("\n[1/4] Loading models...")
        device = 0 if torch.cuda.is_available() else -1

        # Embedding model for retrieval
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Generation pipelines
        self.moderator_pipeline = pipeline("text2text-generation", model='google/flan-t5-base', device=device)
        self.llm_pipeline = pipeline("text2text-generation", model='google/flan-t5-large', device=device)

        # Sentiment analysis
        self.sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

        print("All models loaded successfully.")

    def _build_knowledge_base(self):
        """
        Loads dataset, chunks it, builds FAISS index with normalized embeddings,
        and prepares optional zero-shot classifier.
        """
        print("\n[2/4] Preparing Knowledge Base...")
        try:
            dataset = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")
            raw_docs = [item for item in dataset['answer'] if item and item.strip()]
            self.knowledge_base = []
            for doc in raw_docs:
                self.knowledge_base.extend(doc.split('\n\n'))
            print(f"Successfully loaded and chunked {len(raw_docs)} documents into {len(self.knowledge_base)} chunks.")
        except Exception as e:
            print(f"Failed to load dataset. Using fallback. Error: {e}")
            self.knowledge_base = [
                "You can update your payment method by going to the 'Billing' section in your account settings. All payment information is encrypted and processed securely over an SSL connection.",
                "To check your order status, please log in to your account and navigate to the 'My Orders' page.",
                "I am very sorry to hear your package has not arrived. Please provide your order number so I can investigate.",
            ]
            print(f"Using fallback KB with {len(self.knowledge_base)} documents.")

        print("\n[3/4] Creating embeddings for the knowledge base...")
        raw_embeddings = self.embedding_model.encode(self.knowledge_base, show_progress_bar=True)
        raw_embeddings = np.array(raw_embeddings).astype('float32')

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.kb_embeddings = raw_embeddings / norms

        print("\n[4/4] Setting up FAISS cosine similarity index...")
        d = self.kb_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.kb_embeddings)
        print("FAISS retriever ready.")

        # Optional zero-shot classifier
        try:
            self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                                      device=0 if torch.cuda.is_available() else -1)
            print("Zero-shot classifier loaded.")
        except Exception:
            self.zero_shot = None
            print("Zero-shot classifier unavailable (skipping).")

    def _rewrite_followup(self, query, history, max_new_tokens=64):
        """
        Rewrite a follow-up query into a standalone question.
        - Uses the llm_pipeline but falls back to a heuristic if the rewrite equals
          the previous user message (which indicates a bad rewrite).
        - Always returns a non-empty string.
        """
        query = query.strip()
        if not history:
            return query

        # Build compact history showing only the last user message and assistant reply
        # (keeps prompt short and focused)
        last_turn = history[-1]
        last_user = last_turn.get('user', '').strip()
        last_assistant = last_turn.get('assistant', '').strip()

        rewrite_prompt = f"""
        Given the following short chat history and a follow-up question, rewrite the follow-up
        question as a single, self-contained question that requires no prior context.
        Return ONLY the rewritten question (no explanation, no punctuation at the end beyond normal).

        Chat history:
        User: {last_user}
        Assistant: {last_assistant}

        Follow-up: {query}

        Standalone question:
        """
        try:
            out = self.llm_pipeline(rewrite_prompt,
                                     max_new_tokens=max_new_tokens,
                                     num_beams=4,
                                     do_sample=False)[0]['generated_text'].strip()
        except Exception as e:
            # If model fails, fallback to simple heuristic
            out = ""

        # Basic sanity checks and fallback:
        #  - if the rewrite is empty or exactly equals the last user question, do heuristic
        #  - if rewrite equals last_user (ignore case & punctuation), fallback
        def norm(s): return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()
        if not out or norm(out) == norm(last_user):
            # Heuristic: attach the last user question as referent to the follow-up
            # e.g., "Is that process secure?" -> "Is that process secure? (Referring to: How do I change my payment method?)"
            if last_user:
                out = f"{query} (Referring to: {last_user})"
            else:
                out = query

        return out


    def _is_query_on_topic(self,
                           query,
                           allowed_topics=None,
                           similarity_threshold=0.44,   # lowered default
                           top_k=5,
                           use_zero_shot=True,
                           debug=True):
        """
        Robust on-topic detector.

        Combines:
         - embedding best cosine similarity (top-1)
         - mean cosine similarity of top_k
         - zero-shot 'off-topic' probability -> converted to on-topic prob
         - simple keyword whitelist fallback

        Returns True if combined_score >= similarity_threshold.
        """
        if allowed_topics is None:
            allowed_topics = ['billing', 'orders', 'shipping', 'account', 'product issue', 'returns', 'security']

        q = query.strip().lower()
        if len(q) == 0:
            return False

        # Quick keyword whitelist: immediate accept if contains explicit intent words
        keywords = ['payment', 'pay', 'card', 'invoice', 'order', 'tracking', 'shipment', 'ship', 'shipping',
                    'password', 'login', 'signin', 'account', 'refund', 'return', 'cancel', 'billing', 'subscribe',
                    'subscription', 'charge', 'charged', 'security']
        for kw in keywords:
            if kw in q:
                if debug:
                    print(f"[Safeguard-kw] Keyword '{kw}' matched -> ACCEPT")
                return True

        # Embedding-based scores
        q_emb = self.embedding_model.encode([query])
        q_emb = np.array(q_emb).astype('float32')
        q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)

        # Search top_k (IndexFlatIP stored normalized vectors)
        D, I = self.index.search(q_emb, top_k)  # D: inner-products ~ cosine
        d_list = [float(x) for x in D[0] if x is not None]
        if len(d_list) == 0:
            if debug:
                print("[Safeguard] No neighbors returned by FAISS.")
            embedding_best = 0.0
            embedding_mean = 0.0
        else:
            embedding_best = d_list[0]
            embedding_mean = float(sum(d_list) / len(d_list))

        if debug:
            print(f"[Safeguard] embedding_best={embedding_best:.4f}, embedding_mean(top{top_k})={embedding_mean:.4f}")

        # Zero-shot: compute probability of being on-topic = 1 - P(off-topic)
        zs_on_prob = 0.0
        if use_zero_shot and self.zero_shot is not None:
            try:
                candidate_labels = allowed_topics + ["off-topic"]
                zs = self.zero_shot(query, candidate_labels, multi_label=False)
                # find index of 'off-topic' label and its score
                off_idx = zs['labels'].index('off-topic') if 'off-topic' in zs['labels'] else None
                off_score = 0.0
                if off_idx is not None:
                    off_score = float(zs['scores'][off_idx])
                zs_on_prob = 1.0 - off_score
                if debug:
                    print(f"[Safeguard] zero-shot off-topic_score={off_score:.3f} -> on_prob={zs_on_prob:.3f} (top_label='{zs['labels'][0]}')")
            except Exception as e:
                if debug:
                    print(f"[Safeguard] zero-shot failed: {e}")
                zs_on_prob = 0.0

        # Combine signals with weights (tune these if needed)
        # We give embedding_best the most weight, embedding_mean helps stability, zs_on_prob is supportive.
        w_best = 0.55
        w_mean = 0.25
        w_zs = 0.20
        combined_score = (w_best * max(0.0, embedding_best) +
                          w_mean * max(0.0, embedding_mean) +
                          w_zs * max(0.0, zs_on_prob))

        if debug:
            print(f"[Safeguard] combined_score={combined_score:.4f}, threshold={similarity_threshold}")

        return combined_score >= similarity_threshold

    def _retrieve_context(self, query, k=3):
        """
        Retrieves the top-k most relevant chunks from the knowledge base
        based on cosine similarity of sentence embeddings.
        """
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(np.array(query_embedding).astype("float32"), k)
        retrieved_docs = [self.knowledge_base[i] for i in indices[0]]
        context = "\n\n".join(retrieved_docs)
        return context

    def get_rag_response(self, query, history, k=3):
        """
        Generates a RAG-based response with safeguards. Uses a robust rewrite-first flow.
        """
        print(f"\nProcessing query: '{query}'")

        # Build chat history text for debug and rewriting
        history_string = "".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}\n" for turn in history])

        # 1) Rewrite follow-up into standalone question BEFORE the safeguard
        standalone_query = self._rewrite_followup(query, history)
        print(f"Rewritten query for retrieval & safeguard: '{standalone_query}'")

        # 2) Safeguard check on the standalone query
        if not self._is_query_on_topic(standalone_query, similarity_threshold=0.44, top_k=5, use_zero_shot=True):
            return ("I'm sorry — I can only assist with customer-service related questions "
                    "like billing, orders, shipping, or account issues. Could you rephrase your question?")

        # 3) Sentiment (optional; can be done earlier if you want)
        sentiment = self.sentiment_classifier(standalone_query)[0]['label']
        print(f"Detected Sentiment: {sentiment}")

        # 4) Retrieve context using the standalone query
        context = self._retrieve_context(standalone_query, k=k)

        # 5) Persona and final prompt (use standalone query; forbid echo)
        if sentiment == 'NEGATIVE':
            persona = ("You are an exceptionally empathetic and understanding customer support agent. "
                       "Acknowledge frustration, apologize, and provide the next steps clearly.")
        else:
            persona = ("You are a friendly, efficient, and professional customer support agent. "
                       "Provide clear, concise, and helpful answers.")

        prompt = f"""
        {persona}

        Your role is STRICTLY to be a customer support agent.
        Use only the provided context to answer precise customer-support questions.
        If the answer is not in the context, say you don't know and provide a safe next step (e.g., ask for order number).
        Do NOT repeat the question back in your answer. Return a concise answer of 1-3 sentences.

        Context:
        {context}

        Question: {standalone_query}

        Answer:
        """

        start_time = time.time()
        llm_output = self.llm_pipeline(prompt, max_new_tokens=150, num_beams=4, early_stopping=True)
        response = llm_output[0]['generated_text'].strip()
        print(f"LLM Response Time: {time.time() - start_time:.2f}s")

        # Some models sometimes return the question as the output when confused; guard against that:
        if response.lower().startswith(standalone_query.lower()):
            # If it echoed the question, ask the model one more time with an explicit instruction
            retry_prompt = prompt + "\n(Do NOT repeat the question; give the answer only.)\nAnswer:"
            llm_output = self.llm_pipeline(retry_prompt, max_new_tokens=150, num_beams=4, early_stopping=True)
            response = llm_output[0]['generated_text'].strip()

        return response



# --- Terminal Demo ---
if __name__ == "__main__":
    agent = CustomerServiceAgent()
    conversation_history = []

    print("\n--- Testing  ---")
    query1 = "how do i change my password?"
    response1 = agent.get_rag_response(query1, conversation_history)
    conversation_history.append({'user': query1, 'assistant': response1})
    print(f"\nUser: {query1}\nAgent: {response1}")

    query2 = "my package never arrived."
    response2 = agent.get_rag_response(query2, conversation_history)
    conversation_history.append({'user': query2, 'assistant': response2})
    print(f"\nUser: {query2}\nAgent: {response2}")

    print("\n--- Testing Safeguard (Off-topic) ---")
    query3 = "What's the best recipe for lasagna?"
    response3 = agent.get_rag_response(query3, [])
    print(f"\nUser: {query3}\nAgent: {response3}")

    print("\n--- Demo Complete ---")