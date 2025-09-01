import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentRecommender:
    def __init__(self, products_csv="data/products.csv"):
        self.df = pd.read_csv(products_csv)
        # combine textual fields
        self.df['meta'] = (self.df['name'].fillna('') + ' ' +
                           self.df['category'].fillna('') + ' ' +
                           self.df['description'].fillna(''))
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['meta'])

    def recommend_by_product_id(self, product_id, top_k=5):
        if product_id not in self.df['product_id'].values:
            return []
        idx = int(self.df.index[self.df['product_id']==product_id][0])
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sims[idx] = -1  # ignore itself
        top_idxs = np.argsort(-sims)[:top_k]
        return self.df.iloc[top_idxs][['product_id','name','category','price']].to_dict(orient='records')

    def recommend_by_query(self, text, top_k=5):
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idxs = np.argsort(-sims)[:top_k]
        return self.df.iloc[top_idxs][['product_id','name','category','price']].to_dict(orient='records')

# quick test
if __name__ == "__main__":
    r = ContentRecommender(r"C:\Users\Alawakey\Desktop\ai_customer_support\data\products.csv")
    print(r.recommend_by_query("comfortable audio headphones", top_k=3))
