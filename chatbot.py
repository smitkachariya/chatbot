import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import random

# 1. Load MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Mood detection keywords
MOOD_KEYWORDS = {
    'sad': ['sad', 'depressed', 'down', 'upset', 'crying', 'heartbroken', 'lonely', 'blue', 'unhappy', 'miserable'],
    'stressed': ['stressed', 'anxious', 'overwhelmed', 'pressure', 'panic', 'worried', 'nervous', 'tense'],
    'angry': ['angry', 'mad', 'frustrated', 'annoyed', 'pissed', 'furious', 'hate', 'irritated'],
    'happy': ['happy', 'great', 'excited', 'awesome', 'wonderful', 'amazing', 'fantastic', 'joy', 'good mood'],
    'tired': ['tired', 'exhausted', 'drained', 'burnt out', 'sleepy', 'fatigued', 'weary'],
    'bored': ['bored', 'nothing to do', 'entertain me', 'dull']
}

# 3. Dynamic jokes library
JOKES = {
    'general': [
        "Why don't scientists trust atoms? Because they make up everything! 😄",
        "Why did the scarecrow win an award? He was outstanding in his field! 🌾",
        "What do you call a bear with no teeth? A gummy bear! 🐻",
        "Why don't eggs tell jokes? They'd crack each other up! 🥚",
        "What do you call a fake noodle? An impasta! 🍝",
        "Why did the bicycle fall over? Because it was two tired! 🚲",
        "What's orange and sounds like a parrot? A carrot! 🥕",
        "Why did the math book look sad? Because it had too many problems! 📕",
        "What do you call cheese that isn't yours? Nacho cheese! 🧀",
        "Why can't you give Elsa a balloon? Because she'll let it go! 🎈"
    ],
    'programmer': [
        "Why do programmers prefer dark mode? Because light attracts bugs! 💻",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem! 💡",
        "Why do Java developers wear glasses? Because they can't C#! 👓",
        "What's a programmer's favorite hangout? The Foo Bar! 🍺",
        "Why did the programmer quit? Because they didn't get arrays! 📊"
    ],
    'uplifting': [
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus! 🇨🇭",
        "Why don't scientists trust stairs? Because they're always up to something! 🪜",
        "What do you call a happy cowboy? A jolly rancher! 🤠",
        "Why did the cookie go to the doctor? Because it felt crumbly! 🍪",
        "What did one wall say to the other? I'll meet you at the corner! 🧱"
    ],
    'silly': [
        "What do you call a sleeping bull? A bulldozer! 🐂",
        "Why don't oysters donate to charity? Because they're shellfish! 🦪",
        "What did the ocean say to the beach? Nothing, it just waved! 🌊",
        "Why did the tomato turn red? Because it saw the salad dressing! 🍅",
        "What do you call a dinosaur that crashes cars? Tyrannosaurus Wrecks! 🦕"
    ]
}

# 4. Mood-based responses
MOOD_RESPONSES = {
    'sad': [
        "I'm here for you 💙. Remember, tough times don't last, but tough people do!",
        "It's okay to feel sad. Want to hear something uplifting? 🌟",
        "Sending you virtual hugs 🤗. What would help you feel better right now?"
    ],
    'stressed': [
        "Take a deep breath 😌. You don't have to handle everything at once!",
        "Stress is tough! Remember: you're stronger than you think 💪",
        "Let's tackle this together. What's one small thing you can do right now?"
    ],
    'angry': [
        "It's okay to be angry. Take a moment to breathe 🌬️",
        "I hear you! Sometimes we need to vent. Want to talk about it?",
        "Anger is natural. What would help you cool down?"
    ],
    'happy': [
        "That's wonderful! Your happiness is contagious! ✨😊",
        "I'm so glad you're feeling great! Keep that positive energy! 🌟",
        "Awesome! Love to hear you're in a good mood! 🎉"
    ],
    'tired': [
        "You've been working hard! Rest is important. Can you take a break? 😴",
        "Exhaustion is real. Your body needs care. Please rest if you can 💙",
        "When you're tired, it's time to recharge. You deserve rest! 🌙"
    ],
    'bored': [
        "Let's fix that boredom! Want a joke, fun fact, or activity suggestion? 🎨",
        "Boredom is opportunity! Want to hear something interesting? 🌟",
        "I can entertain you! How about a joke or riddle? 🎭"
    ]
}

# 4. Text preprocessing function
def preprocess_text(text):
    """Clean and normalize text for better matching"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?\!\.\,\']', '', text)
    return text

# 5. Detect mood from user input
def detect_mood(text):
    """Detect user's emotional state from their message"""
    text_lower = text.lower()
    detected_moods = []
    
    for mood, keywords in MOOD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_moods.append(mood)
                break
    
    return detected_moods

# 6. Get dynamic joke based on context
def get_dynamic_joke(mood=None):
    """Return a joke appropriate for the user's mood"""
    if mood == 'sad' or mood == 'stressed':
        # Uplifting jokes for sad/stressed users
        return random.choice(JOKES['uplifting'])
    elif mood == 'bored':
        # Mix of all jokes for bored users
        all_jokes = JOKES['general'] + JOKES['silly'] + JOKES['programmer']
        return random.choice(all_jokes)
    else:
        # Random joke from general collection
        return random.choice(JOKES['general'])

# 7. Get mood-based response
def get_mood_response(moods):
    """Return an empathetic response based on detected mood"""
    if not moods:
        return None
    
    # Prioritize negative emotions for support
    priority = ['sad', 'stressed', 'angry', 'tired', 'bored', 'happy']
    for mood in priority:
        if mood in moods:
            return random.choice(MOOD_RESPONSES[mood])
    
    return None

# 8. Get context-aware response prefix
def get_context_aware_prefix(current_mood):
    """Add empathetic context if user is in a particular mood"""
    prefixes = {
        'sad': [
            "I know you're feeling down right now. ",
            "I'm still here for you. ",
            "Sending you support. "
        ],
        'stressed': [
            "I know things feel overwhelming. ",
            "Take it one step at a time. ",
            "Remember to breathe. "
        ],
        'angry': [
            "I understand you're upset. ",
            "Take your time to cool down. ",
        ],
        'tired': [
            "I know you're exhausted. ",
            "Make sure to rest when you can. "
        ]
    }
    
    if current_mood in prefixes:
        return random.choice(prefixes[current_mood])
    return ""

# 9. Load question-answer dataset
def load_knowledge(file_path):
    questions = []
    answers = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                q, a = line.strip().split("|", 1)
                questions.append(preprocess_text(q))
                answers.append(a.strip())

    return questions, answers

questions, answers = load_knowledge("data/knowledge_emotional.txt")

# 8. Create embeddings for questions
question_embeddings = model.encode(questions)

# 9. Create FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

# 10. Conversation history with emotional context
conversation_history = []
current_user_mood = None  # Track user's current emotional state
mood_persistence_count = 0  # How many turns to remember mood

print("🤖 Emotional Intelligence Chatbot is ready!")
print("💙 I can detect your mood and provide support, motivation, jokes, and more!")
print("Type 'exit' to quit.\n")

# 11. Chat loop with mood detection and context awareness
while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("Bot: Take care! Remember, you're awesome! Come back anytime! 👋✨")
        break

    # Detect mood from user input
    detected_moods = detect_mood(query)
    
    # Update current mood if a strong emotion is detected
    if detected_moods:
        # Prioritize negative emotions for support
        priority = ['sad', 'stressed', 'angry', 'tired']
        for mood in priority:
            if mood in detected_moods:
                current_user_mood = mood
                mood_persistence_count = 5  # Remember for next 5 messages
                break
        else:
            # If only positive moods detected
            if 'happy' in detected_moods:
                current_user_mood = 'happy'
                mood_persistence_count = 3
    
    # Decrease mood persistence
    if mood_persistence_count > 0:
        mood_persistence_count -= 1
    else:
        current_user_mood = None
    
    mood_response = get_mood_response(detected_moods)
    
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # Check if user is asking for a joke
    joke_keywords = ['joke', 'funny', 'laugh', 'humor', 'make me smile']
    wants_joke = any(keyword in processed_query for keyword in joke_keywords)
    
    # Get top 3 matches for better accuracy
    query_embedding = model.encode([processed_query])
    distances, indices = index.search(np.array(query_embedding), k=3)

    # Calculate confidence
    best_distance = distances[0][0]
    
    # Dynamic threshold
    if best_distance > 0.8:
        # If mood detected, show empathy first
        if mood_response:
            print(f"Bot: {mood_response}")
        else:
            # Add context if user has ongoing mood
            if current_user_mood:
                context_prefix = get_context_aware_prefix(current_user_mood)
                print(f"Bot: {context_prefix}How can I help you?")
            else:
                print(f"Bot: I'm not sure I understood that, but I'm here to help! 😊")
        
        # Offer a joke if user seems down
        if current_user_mood in ['sad', 'stressed', 'bored']:
            print(f"\n     💡 Would you like to hear a joke to lighten the mood?")
        
        # Show similar questions as suggestions
        if best_distance < 1.5:
            print("\n     You might be interested in:")
            for i in range(min(2, len(indices[0]))):
                if distances[0][i] < 1.5:
                    print(f"     - {questions[indices[0][i]]}")
            print()
    else:
        answer = answers[indices[0][0]]
        
        # Dynamic joke replacement if user wants a joke
        if wants_joke:
            dynamic_joke = get_dynamic_joke(current_user_mood)
            # Replace generic jokes with dynamic ones
            if "why don't scientists trust atoms" in answer.lower() or "joke" in processed_query:
                answer = dynamic_joke
        
        # Add context-aware prefix if user has ongoing emotional state
        context_prefix = ""
        if current_user_mood and not detected_moods:
            context_prefix = get_context_aware_prefix(current_user_mood)
        
        # If new mood detected, show empathy first
        if mood_response and detected_moods:
            print(f"Bot: {mood_response}\n")
            print(f"Bot: {answer}")
        elif context_prefix:
            print(f"Bot: {context_prefix}{answer}")
        else:
            print(f"Bot: {answer}")
        
        # Store in conversation history
        conversation_history.append({
            "query": query,
            "response": answer,
            "detected_moods": detected_moods,
            "persistent_mood": current_user_mood
        })
        
        # Show conversation insights every 5 messages
        if len(conversation_history) % 5 == 0 and current_user_mood:
            print(f"\n💙 I've noticed you've been feeling {current_user_mood}. I'm here to support you!\n")
