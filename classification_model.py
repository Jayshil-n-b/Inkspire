import pickle
import numpy as np
import tensorflow as tf
import transformers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, Dense
from transformers import DistilBertTokenizer, TFAutoModel, AdamWeightDecay
from tensorflow_addons.metrics import F1Score

from scrapper import Scrapper



class Classification_model:
    def __init__(self) -> None:
        self.model = tf.keras.models.load_model('my_model.keras', custom_objects={"TFAutoModel": transformers.TFAutoModel, "TFDistilBertModel": transformers.TFDistilBertModel, "AdamWeightDecay": transformers.AdamWeightDecay, "F1Score": F1Score})
        self.tokenizer = pickle.load(open(b"tokenizer_1.h5","rb"))
        self.max_len = pickle.load(open(b"max_len_1.unknown","rb"))

        self.classes_name = ['ARTS & CULTURE',
        'BUSINESS & FINANCES',
        'COMEDY',
        'CRIME',
        'DIVORCE',
        'EDUCATION',
        'ENTERTAINMENT',
        'ENVIRONMENT',
        'FOOD & DRINK',
        'GROUPS VOICES',
        'HOME & LIVING',
        'IMPACT',
        'MEDIA',
        'MISCELLANEOUS',
        'PARENTING',
        'POLITICS',
        'RELIGION',
        'SCIENCE & TECH',
        'SPORTS',
        'STYLE & BEAUTY',
        'TRAVEL',
        'U.S. NEWS',
        'WEDDINGS',
        'WEIRD NEWS',
        'WELLNESS',
        'WOMEN',
        'WORLD NEWS']
    
    def _tokenizer_preprocessing(self, texts, tokenizer):
        encoded_dict = self.tokenizer.batch_encode_plus(
            texts,
            return_token_type_ids=False,
            pad_to_max_length=True, # the length of all texts will be equal to a text which has the maximum tokens
            max_length=self.max_len
        )
        return np.array(encoded_dict['input_ids']) # convert a list to an array

    def predict_tags(self, article, top_count = 3):
        article_vec = self._tokenizer_preprocessing([article], self.tokenizer)
        article_pred = self.model.predict(article_vec)
        # [[[0, 0.04], [1, 0.06],, [18, 0.81], [26, 0.07]]]
        article_pred_new = article_pred[0]

        article_score_class = []
        for i in range(27):
            article_score_class.append([article_pred_new[i], i])

        article_score_class.sort()
        predicted_tags = []

        count = 0
        while(count < top_count):
            predicted_tags.append(self.classes_name[article_score_class[26 - count][1]])
            count += 1

        return predicted_tags

