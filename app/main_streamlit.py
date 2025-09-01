import streamlit as st
from recommender import ContentRecommender
from faq_bot import FAQBot
from llm_interface import ask_hf
from sentiment import analyze_sentiment
from db_utils import init_db, save_conversation

# Initialize DB and tools
init_db()
rec = ContentRecommender(r"C:\Users\Alawakey\Desktop\ai_customer_support\data\products.csv")
faq = FAQBot(r"C:\Users\Alawakey\Desktop\ai_customer_support\data\faqs.csv")

# App title
st.title("AI Customer Support")

# User inputs
user_id = st.text_input("User ID (optional)", value="guest")
user_input = st.text_input("Type your question here...")

if st.button("Send"):
    # 1) Try FAQ first
    answer, score = faq.get_answer(user_input)
    if answer:
        bot_resp = f"(FAQ) {answer}"
    else:
        # 2) Ask LLM
        bot_resp = ask_hf(user_input)

    # 3) Recommendations - simple keyword trigger
    if any(w in user_input.lower() for w in ["recommend", "suggest", "similar", "product"]):
        recs = rec.recommend_by_query(user_input, top_k=3)
        rec_text = "\n".join([f"- {r['name']} ({r['category']}) ${r['price']}" for r in recs])
        bot_resp += "\n\nRecommended products:\n" + rec_text

    # 4) Sentiment analysis
    sent = analyze_sentiment(user_input)[0]
    st.markdown("**Bot response:**")
    st.write(bot_resp)
    st.markdown("**Sentiment of user message:**")
    st.write(sent)

    # 5) Save conversation
    save_conversation(user_id, user_input, bot_resp, str(sent))
