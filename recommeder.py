from classification_model import Classification_model
from scrapper import Scrapper
from encoder import Encoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class ArticleRecommender:
    def __init__(self) -> None:
        self.article_scrapper = Scrapper()
        self.article_encoder = Encoder()
        self.classification_model = Classification_model()
        self.news_df = pd.read_csv("updated_news_articles.csv") #200000
        self.embeddings = np.load("embeddings2.npy")
        self.predicted_tags = []


    def recommed_top_articles(self, url, num_recommendations = 10):
        print("At the start-----------------------------------")
        print(len(self.news_df.index))
        print(self.embeddings.shape)

        article_title, article_content = self.article_scrapper.scrape_article(url)
        article_text = article_title + " " + article_content
        # print(article_text)
        predicted_tags = self.classification_model.predict_tags(article_text, 3)
        self.predicted_tags = predicted_tags
        # predicted_tags = ["ENVIRONMENT"]

        df = self.news_df[self.news_df['category'].isin([predicted_tags[0]])] #50000

        article_encoding = self.article_encoder.getArticleEncoding(article_text)
        # print(self.embeddings.shape)
        # print(article_encoding.shape)

        # limit = 2
        similarities = []

        print("Going inside loop")
        for ind in df.index:
            # print(df['index'][ind])
            emb = self.embeddings[df['index'][ind]]
            # print(emb.shape)
            # print(article_encoding.shape)

            score = cosine_similarity([article_encoding, emb])[0,1]
            # [1, 0.4]
            # [0.4, 1]
            similarities.append([score, df['index'][ind]])
            # limit -= 1
            # if limit == 0:
            #     break
        print("Out of loop")

        similarities = sorted(similarities ,key=lambda l:l[0], reverse=True)
        result = similarities[:num_recommendations]
        

        top_articles = []
        for r in result:
            curr_article = df[df['index'] == r[1]].to_dict('records')[0]
            curr_article["match"] = r[0]
            top_articles.append(curr_article)


        print("After procedure-----------------------------------")
        print(len(self.news_df.index))
        print(self.embeddings.shape)
        return top_articles
    
    def getArticleTags(self):
        return self.predicted_tags

