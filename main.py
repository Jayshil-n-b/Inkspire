from recommeder import ArticleRecommender
import streamlit as st
from streamlit_card import card

article_recommender = ArticleRecommender()


st.title('Article Recommendation')

scrapedText = "Here comes your article"
url = st.text_input("Article Link", value="")

if st.button("Enter") and len(url) > 0:

    st.empty()

    top_articles = article_recommender.recommed_top_articles(url, 10)

    st.write(f"#### The tags of article are: {article_recommender.predicted_tags}")
    
    for article in top_articles:
        st.write(f"### {article['headline']}")
        st.caption(article["category"])
        st.markdown(f"[{article['link']}]({article['link']})")
        if article['short_description']:
            st.caption(article["short_description"])
        st.write("---------------------------------------")

