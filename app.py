import time
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import datetime
import random


st.set_page_config(page_title="Flight Booking FAQ Bot", page_icon="🛫", layout="centered")


st.title("🛫 Welcome to the Flight Booking FAQ Bot! ✨")


with open('FAQ.json', 'r') as file:
    faq_data = json.load(file)


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
faq_questions = [faq['question'] for faq in faq_data['FAQs']]
faq_embeddings = np.array([model.encode(q) for q in faq_questions])

index = faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(faq_embeddings)


if 'history' not in st.session_state:
    st.session_state['history'] = []


def user_message(message_text, timestamp):
    st.markdown(f"""
    <div style='text-align: right; padding: 5px;'>
        <div style='display: inline-block; background: #00FFFF; padding: 10px 15px; border-radius: 15px; color: black;
        animation: fadeIn 0.5s; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);'>
            {message_text}
            <div style='font-size:10px;color:gray;text-align:right;'>{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def bot_message(message_text, timestamp):
    st.markdown(f"""
    <div style='text-align: left; padding: 5px;'>
        <div style='display: inline-block; background: #007FFF; padding: 10px 15px; border-radius: 15px; color: white;
        animation: fadeIn 0.5s; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);'>
            {message_text}
            <div style='font-size:10px;color:lightgray;text-align:right;'>{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def simulate_typing_effect():
    typing_placeholder = st.empty()
    typing_text = ""
    for _ in range(3):  # simulate 3 dots
        typing_text += "."
        typing_placeholder.markdown(f"🤖 Bot is typing{typing_text}", unsafe_allow_html=True)
        time.sleep(0.4)
    typing_placeholder.empty()

def generate_response(query):
    query_embedding = model.encode(query)
    D, I = index.search(np.array([query_embedding]), k=3)

    simulate_typing_effect()

    confidence_threshold = 0.5
    responses = []

    if D[0][0] < confidence_threshold:
        for idx in I[0]:
            answer = faq_data['FAQs'][int(idx)]['answer']
            responses.append(f"- {answer}")
    else:
        responses.append("🤔 No exact match found! Here are some questions you can ask:")
        suggestions = random.sample(faq_questions, 4)
        for sug in suggestions:
            responses.append(f"- {sug}")

    return responses


st.markdown("### 💬 Quick Questions:")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔑 Forgot Password?"):
        clicked = "I forgot my password, what should I do?"
with col2:
    if st.button("🛫 My Bookings?"):
        clicked = "Where can I find how many flights I have booked?"

col3, col4 = st.columns(2)
with col3:
    if st.button("📜 Previous Flights?"):
        clicked = "How do I see my previous flight bookings?"
with col4:
    if st.button("💰 Commission Details?"):
        clicked = "Where can I find commission details for a booking?"


if 'clicked' in locals():
    now = datetime.datetime.now().strftime("%H:%M")
    st.session_state.history.append({"role": "user", "message": clicked, "time": now})

    responses = generate_response(clicked)
    now = datetime.datetime.now().strftime("%H:%M")
    for res in responses:
        st.session_state.history.append({"role": "bot", "message": res, "time": now})
        time.sleep(0.7)  # small pause between bot responses


user_input = st.text_input("You:", key="user_input_box")
if user_input:
    now = datetime.datetime.now().strftime("%H:%M")
    st.session_state.history.append({"role": "user", "message": user_input, "time": now})

    responses = generate_response(user_input)
    now = datetime.datetime.now().strftime("%H:%M")
    for res in responses:
        st.session_state.history.append({"role": "bot", "message": res, "time": now})
        time.sleep(0.7)  # small pause between bot responses


for chat in st.session_state.history:
    if chat['role'] == 'user':
        user_message(chat['message'], chat['time'])
    else:
        bot_message(chat['message'], chat['time'])


st.markdown("""
<style>
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)
