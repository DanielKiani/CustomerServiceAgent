![Banner](assets/banner.png)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch)](https://pytorch.org/)[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# 🤖 Advanced Customer Service Agent

An intelligent customer service agent built with a Retrieval-Augmented Generation (RAG) pipeline Made from scratch and also with Langchain. This agent understands user sentiment, retrieves information from a knowledge base, and provides empathetic, context-aware responses. It features a robust, multi-layered safeguard system to ensure conversations remain on-topic and safe.

The live Gradio demo is hosted on Hugging Face Spaces: **[🚀 View Demo Here](https://huggingface.co/spaces/Deathshot78/CustomerServiceAgent)**

![Gradio](assets/gradio.png)

---

## 📋 Table of Contents

- [📖 About The Project](#-about-the-project)
- [✨ Features](#-features)
- [🧠 Project Journey & Key Learnings](#-project-journey--key-learnings)
- [🛠️ Final Architecture & Tech Stack](#️-final-architecture--tech-stack)
- [🔮 Future Improvements](#-future-improvements)
- [🚀 Getting Started](#-getting-started)

---

## 📖 About The Project

This project chronicles the end-to-end development of an AI customer service agent, from a simple prototype to a production-ready application with advanced safeguards and with Langchain and from scratch implementations. The agent's core is a RAG pipeline that answers queries based on a predefined knowledge base. The final version integrates a custom, multi-signal moderation system to handle off-topic questions and a dynamic prompting strategy to adapt its tone based on user sentiment.

---

## ✨ Features

- **🛡️ Advanced Safeguards**: A custom, multi-signal moderation system rejects off-topic queries by combining keyword matching, embedding similarity, and zero-shot classification.
- **🧠 Conversation Memory**: Remembers previous turns to understand context and handle follow-up questions effectively.
- **😠 Dynamic Persona**: Detects user sentiment (`Positive`/`Negative`) and dynamically adjusts its persona in the prompt to be more helpful or empathetic.
- **📚 Retrieval-Augmented Generation (RAG)**: Retrieves relevant "chunks" of information from a FAISS vector database to provide accurate, knowledge-based answers.
- **🔊 Text-to-Speech**: Can read its responses aloud for a complete voice-enabled experience.
- **🌐 Interactive UI**: Built with Gradio for an easy-to-use web interface.

---

## 🧠 Project Journey & Key Learnings

This project evolved significantly, with each phase revealing new challenges and leading to more sophisticated solutions.

#### 1. The Quality vs. Speed Dilemma
The initial prototype used `google/flan-t5-base` for fast responses (~4 seconds). However, it struggled to follow persona instructions, often giving blunt or unhelpful answers to frustrated users. We benchmarked this against `google/flan-t5-large`. While significantly slower (~20 seconds on a CPU), the larger model's ability to adopt an empathetic persona was a non-negotiable requirement for a customer service agent. **Key Learning:** For user-facing applications, response quality and the ability to follow nuanced instructions are often more important than raw speed.

#### 2. The Safeguard Challenge
Ensuring the agent stayed on-topic was the most critical challenge.
- **Initial Failure:** A simple, prompt-based moderator using `flan-t5-base` proved unreliable. It failed to understand context-dependent follow-up questions and was easily fooled by well-formed but irrelevant queries (e.g., "What's the recipe for lasagna?").
- **The Breakthrough - A Multi-Signal Approach:** The final solution was a "Defense in Depth" strategy implemented in a single function. Instead of relying on one signal, our safeguard combines three:
  1.  **Keyword Heuristics:** A fast check for obvious on-topic words.
  2.  **Embedding Similarity:** Measures the semantic relevance of a query against the entire knowledge base.
  3.  **Zero-Shot Classification:** Uses a dedicated classifier (`facebook/bart-large-mnli`) to explicitly categorize the query into allowed topics or "off-topic."
- **Final Logic:** By combining these signals with weighted scores, we created a robust and nuanced gatekeeper that successfully rejects irrelevant queries while understanding legitimate follow-ups.

#### 3. Refactoring for Production
With the core logic proven, the final step was to refactor the project for maintainability and scalability. I explored both a from-scratch implementation and a version using the **LangChain** framework. The final version combines the best of both worlds: it uses LangChain's powerful components (like `ConversationalRetrievalChain`) but replaces its default moderation with our superior, custom-built multi-signal safeguard.

#### 4. The "Garbage-In, Garbage-Out" Principle: Knowledge Base is King
Even with advanced safeguards and a capable LLM, the agent's performance is fundamentally limited by the quality of its knowledge base. We observed several "retrieval failures" where the agent gave factually incorrect or irrelevant answers. For example, when asked for the information needed to find a lost package, the retriever found a document about returning a *wrong item* because it was the most semantically similar text in the generic FAQ dataset. The LLM then correctly answered based on this faulty context. **Key Learning:** A RAG system is only as good as its knowledge. The most significant improvement for a production system is not a better model, but a highly curated, accurate, and specific knowledge base tailored to the agent's exact domain.

---

## 🛠️ Final Architecture & Tech Stack

The final architecture is a robust pipeline with a pre-processing safeguard gate.

1.  **User Query**: The user asks a question.
2.  **🛡️ Safeguard Gate**: The query is first sent to our multi-signal moderator. If it's off-topic, the process stops and a polite refusal is returned.
3.  **Sentiment Analysis**: If the query is on-topic, its sentiment is analyzed.
4.  **Conversational Rewriting**: Follow-up questions are rewritten into standalone queries for better retrieval.
5.  **RAG Pipeline**: The standalone query is used to retrieve context from the FAISS index.
6.  **Dynamic Prompting**: A prompt is constructed with the persona, guardrails, and retrieved context.
7.  **LLM Response Generation**: The prompt is sent to `google/flan-t5-large` to generate the final answer.

| Component | Model / Library |
| :--- | :--- |
| **Orchestration** | LangChain / Custom Python |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Response Generation** | `google/flan-t5-large` |
| **Safeguard (Moderation)** | Custom multi-signal logic using `facebook/bart-large-mnli` |
| **Vector Store** | `faiss-cpu` |
| **User Interface** | `gradio` |
| **Text-to-Speech** | `gTTS` |

---

## 🔮 Future Improvements

- **Fine-Tune a Specialized Moderator**: For ultimate accuracy, the zero-shot classifier in the safeguard could be replaced with a smaller model (like DistilBERT) fine-tuned on thousands of company-specific on-topic/off-topic examples.
- **Output Moderation**: Add a final check on the agent's response *before* it's sent to the user to scan for PII, harmful language, or factual inconsistencies against the source context.
- **Customize the Knowledge Base**: Replace the generic FAQ dataset with a company's internal documentation and past support tickets to create a highly specialized and valuable internal tool.
- **🐳 Dockerize for Deployment**: Containerize the application using Docker for consistent and scalable deployment across different environments.

---

## 🚀 Getting Started

Follow these steps to get the agent running locally.
**Note :** you can find the code for the from scratch implementation and the Langchain version in the scripts folder. you can run the Gradio app for the langchain version locally as i only have the from scratch implementaino up in the huggingface spaces.

**From scratch implementation scripts:** `agent.py` , `app.py`
**Langchain version scripts:** `agent_langchain.py` , `app_langchain.py`

### Prerequisites
You need to have Python 3.8+ and Git installed.

### Installation & Usage
1.  **Clone the repository:**
    ```sh
    git clone https://github.com/DanielKiani/CustomerServiceAgent
    cd CustomerServiceAgent
    ```

2.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the terminal-based demo (optional):**
    To see the core agent logic and debug output in your terminal, run `agent.py` or `agent_langchain.py`.
    ```sh
    python agent_langchain.py
    ```

4.  **Launch the Gradio Web App:**
    To start the interactive user interface, run `app.py` or `app_langchain.py`.
    ```sh
    python app_langchain.py
    ```
    This will print a local URL in your terminal. Open it in your browser to interact with the agent.