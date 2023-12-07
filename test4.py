import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("article_data.csv")
sentence_embeddings = np.load("embeddings.npy")


def brand_inference(user_input,sentence_embeddings):
    user_input = re.sub('[^A-Za-z0-9]+', ' ', user_input)
    
    user_input_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_input_embedding, sentence_embeddings)

    num_recommendations = 10
    top_indices = similarities.argsort()[0][::-1][:num_recommendations]

    recommended_products = []
    for idx in top_indices:
        recommended_products.append(df['headline'][idx])
    print(recommended_products)

# input_list = ["bluetooth headphones wired and wireless","asics running shoes","mast & harbour full sleeve ","mast & harbour womens","allure auto car mat"]
input_list = ['''Special session: Modi introduces women's bill in new India parliament The Indian government has introduced a bill guaranteeing a third of seats for women in the lower house of parliament and state assemblies. The contentious bill, first proposed in 1996, has been pending for decades amid opposition from some political parties. Its revival is expected to boost the governing Bharatiya Janata Party's fortunes in general elections next May. The bill was tabled at the new Indian parliament's first session and is still some way from becoming law. It would require the approval of both houses of parliament and a majority of state legislatures, as well as the Indian president's signature. Reported plans to increase the overall  number of constituencies could further complicate its implementation. In his opening speech at the new parliament building, Mr Modi praised the proposed legislation and said it was a special moment for the country.  "The world understands that only talking of women-led development is not enough. This is a positive step taken on that front," he told politicians as he appealed to them to support the bill. The PM also took a swipe at the opposition and said that the previous Congress party-led governments had failed to clear the bill when they were in power.   "There have been discussions around women's reservations for years. We can say with pride that we have scripted history," he said.  Mr Modi inaugurated the new parliament building in May but no business was held there until now. He called a five-day special session which began on Monday but the first day's sitting was held in the old parliament building. On Tuesday morning, members from both houses assembled for a photo session at the old building, followed by an event commemorating parliament's legacy in the Central Hall of the British-era building. They then moved to the new parliament as the office of the lower house of parliament officially designated it as the Parliament House of India.  The proceedings are being held amid criticism from opposition leaders who claim that the government has not disclosed all the business that could come up during the week. According to the government, eight bills have been listed for discussion during the session - but this agenda could be changed or expanded during the course of the week. The new parliament building is part of the government's ambitious Central Vista project in Delhi to replace colonial-era government buildings. Built in front of the old parliament, the new four-storey building - constructed at an estimated cost of 9.7bn rupees ($117m; £94m) - is much bigger and has the capacity to seat 1,272 MPs. The Lok Sabha chamber, which will seat the lower house of the parliament, is designed in the likeness of a peacock, India's national bird. The Rajya Sabha chamber, which will seat the upper house, is designed to resemble the lotus - India's national flower and also Mr Modi's Bharatiya Janata Party's election symbol. The current parliament building will be converted into a museum. BBC News India is now on YouTube. Click here to subscribe and watch our documentaries, explainers and features.''']

for i in input_list:
    brand_inference(i,sentence_embeddings)
    print()