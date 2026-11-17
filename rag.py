
import requests
import numpy as np
from pathlib import path

# Ollama configuration
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


# 1. Generate embeddings
def embed(texts):
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )

    r.raise_for_status()

    return np.array(r.json()["embeddings"])


# 2. Ask Ollama LLM
def ask_llm(prompt):
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        timeout=300
    )

    r.raise_for_status()

    return r.json()["message"]["content"]


# 3. Load documents
chunks = []

docs_folder = Path("docs")

if not docs_folder.exists():
    raise FileNotFoundError("Create a docs folder and add your .md files.")

for file in docs_folder.glob("*.md"):
    text = file.read_text(encoding="utf-8")

    chunks += [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

if not chunks:
    raise ValueError("No Markdown documents found in the docs folder.")

print(f"Loaded {len(chunks)} chunks.")


# 4. Generate document embeddings
chunk_vectors = embed(chunks)

print(f"Indexed {len(chunks)} chunks successfully.")


# 5. Retrieve relevant documents
def retrieve(question, k=4):

    q = embed([question])[0]

    scores = (chunk_vectors @ q) / (
        np.linalg.norm(chunk_vectors, axis=1) *
        np.linalg.norm(q)
    )

    top_indices = np.argsort(scores)[::-1][:k]

    return [chunks[i] for i in top_indices]


# 6. Chatbot using while loop
def main():

    print("\n===== EMPLOYEE HANDBOOK RAG CHATBOT =====")

    while True:

        question = input("\nAsk your question : ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            # Retrieve relevant paragraphs
            results = retrieve(question, k=4)

            # Combine retrieved context
            context = "\n\n".join(results)

            # Create RAG prompt
            prompt = f"""
You are an employee handbook assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say: "This information is not available in the employee handbook."

Context:
{context}

Question:
{question}

Answer:
"""

            # Get answer from Ollama
            answer = ask_llm(prompt)

            print("\n===== AI RESPONSE =====")
            print(answer)

        except requests.exceptions.RequestException as e:
            print(f"Ollama connection error: {e}")

        except Exception as e:
            print(f"Error: {e}")


# 7. Run application
if __name__ == "__main__":
    main()
