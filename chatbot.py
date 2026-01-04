import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load question-answer dataset
def load_knowledge(file_path):
    questions = []
    answers = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                q, a = line.strip().split("|")
                questions.append(q.strip().lower())
                answers.append(a.strip())

    return questions, answers

questions, answers = load_knowledge("data/knowledge.txt")

# 3. Create embeddings ONLY for questions
question_embeddings = model.encode(questions)

# 4. Create FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

print("🤖 MiniLM Chatbot is ready! Type 'exit' to quit.\n")

# 5. Chat loop
while True:
    query = input("You: ").lower()

    if query == "exit":
        print("Bot: Goodbye 👋")
        break

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), 1)

    # Confidence threshold (IMPORTANT)
    if distances[0][0] > 1.0:
        print("Bot: Sorry, I don’t understand that 😕")
    else:
        answer = answers[indices[0][0]]
        print("Bot:", answer)
