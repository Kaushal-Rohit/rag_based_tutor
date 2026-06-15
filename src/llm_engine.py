"""
LLM Engine Module
=================
Local LLM integration via Ollama API with sentiment-aware dynamic prompting.
Includes conversation history tracking and adaptive retrieval depth.
"""

import requests
from textblob import TextBlob


class ConversationManager:
    """
    Tracks conversation history and computes rolling sentiment.

    Maintains a sliding window of the last N interactions.
    Uses TextBlob polarity to classify user state as confused, neutral, or clear.
    """

    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []

    def analyze_sentiment(self, text):
        """Return polarity score: -1.0 (negative/confused) to 1.0 (positive/clear)."""
        return TextBlob(text).sentiment.polarity

    def add_turn(self, query, answer):
        """Record a query-answer pair with its sentiment score."""
        sentiment = self.analyze_sentiment(query)
        self.history.append({
            "query": query,
            "answer": answer,
            "sentiment": sentiment,
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_average_sentiment(self, current_query=None):
        """Compute average sentiment across history and optionally the current query."""
        scores = [turn["sentiment"] for turn in self.history]
        if current_query:
            scores.append(self.analyze_sentiment(current_query))
        return sum(scores) / len(scores) if scores else 0.0

    def get_formatted_history(self):
        """Format conversation history as a string for LLM prompt injection."""
        if not self.history:
            return "No previous conversation history."
        lines = []
        for i, turn in enumerate(self.history):
            lines.append(f"User (Turn {i + 1}): {turn['query']}")
            lines.append(f"AI (Turn {i + 1}): {turn['answer']}\n")
        return "\n".join(lines).strip()


class LocalLLMEngine:
    """
    Interface to a local Ollama instance.

    Sends structured prompts to the Ollama /api/generate endpoint
    with configurable system instructions.
    """

    def __init__(self, model_name="llama3:latest", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def check_connection(self):
        """Verify Ollama server is reachable."""
        try:
            return requests.get(self.base_url).status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def generate(self, prompt, system_instruction=""):
        """Send prompt to Ollama and return the generated text."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_instruction,
            "stream": False,
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            return f"[ERROR] LLM generation failed: {e}"


class DynamicRAGPipeline:
    """
    End-to-end RAG pipeline with sentiment-driven adaptive behavior.

    Behavior modes:
        Confused (sentiment < -0.1): Increases retrieval depth (k+3), detailed responses.
        Clear    (sentiment > 0.3):  Decreases retrieval depth (k-2), concise responses.
        Neutral  (otherwise):        Standard retrieval depth, balanced responses.
    """

    def __init__(self, collection, embedding_model, llm_engine, base_k=5):
        self.collection = collection
        self.embedding_model = embedding_model
        self.llm_engine = llm_engine
        self.base_k = base_k
        self.convo_manager = ConversationManager(max_history=5)

    def _retrieve_chunks(self, query, k, filters=None):
        """Fetch top-k chunks from ChromaDB with optional metadata filtering."""
        query_embedding = self.embedding_model.encode([query]).tolist()

        where_filter = None
        if filters:
            conditions = [{key: {"$eq": value}} for key, value in filters.items()]
            where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_filter,
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return results["documents"][0]

    def chat(self, query, filters=None):
        """
        Process a user query through the full adaptive RAG pipeline.

        Steps:
            1. Compute rolling sentiment from conversation history.
            2. Determine retrieval depth (k) and system instruction based on sentiment.
            3. Retrieve context chunks from ChromaDB.
            4. Construct prompt with history + context + query.
            5. Generate response via local LLM.
            6. Update conversation history.
        """
        avg_sentiment = self.convo_manager.get_average_sentiment(query)

        if avg_sentiment < -0.1:
            k = self.base_k + 3
            system_instruction = (
                "You are a highly patient and detailed tutor. The user seems confused or is struggling "
                "to understand. Provide a very detailed, step-by-step explanation. Break down complex "
                "concepts simply. Use the provided context to answer accurately."
            )
            state_msg = "User is confused -> Increasing K, providing detailed response."

        elif avg_sentiment > 0.3:
            k = max(2, self.base_k - 2)
            system_instruction = (
                "You are an expert tutor. The user is understanding the material well. "
                "Provide highly precise, accurate, and concise answers without unnecessary fluff. "
                "Use the provided context to answer."
            )
            state_msg = "User understands -> Decreasing K, providing precise response."

        else:
            k = self.base_k
            system_instruction = (
                "You are a helpful tutor. Use the provided context to answer the user's question clearly. "
                "If the context doesn't contain the answer, say you don't know based on the book."
            )
            state_msg = "User is neutral -> Standard K, standard response."

        print(f"[LOG] Avg Sentiment: {avg_sentiment:.2f} | {state_msg} (k={k})")

        chunks = self._retrieve_chunks(query, k, filters)
        context_str = "\n\n---\n\n".join(chunks)
        history_str = self.convo_manager.get_formatted_history()

        prompt = (
            f"**Conversation History (Last 5 turns):**\n{history_str}\n\n"
            f"**Retrieved Knowledge Chunks:**\n{context_str}\n\n"
            f"**Current User Query:**\n{query}\n\n"
            f"**Your Answer:**\n"
        )

        print("[LOG] Generating response via Ollama...")
        response = self.llm_engine.generate(prompt, system_instruction)
        self.convo_manager.add_turn(query, response)

        return response
