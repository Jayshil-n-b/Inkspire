from sentence_transformers import SentenceTransformer, util
import re

class Encoder:
    def __init__(self) -> None:
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def getArticleEncoding(self, article):
        article_input = re.sub('[^A-Za-z0-9]+', ' ', article)
        article_input_embedding = self.encoder.encode([article_input])

        return article_input_embedding[0]