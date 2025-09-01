import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQBot:
    def __init__(self, faqs_csv=r"C:\Users\Alawakey\Desktop\ai_customer_support\data\faqs.csv"):
        self.df = pd.read_csv(faqs_csv)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf = self.vectorizer.fit_transform(self.df['question'].astype(str))

    def get_answer(self, user_question, threshold=0.45):
        q_vec = self.vectorizer.transform([user_question])
        sims = cosine_similarity(q_vec, self.tfidf).flatten()
        best_idx = sims.argmax()
        if sims[best_idx] >= threshold:
            return self.df.iloc[best_idx]['answer'], float(sims[best_idx])
        return None, float(sims[best_idx])
