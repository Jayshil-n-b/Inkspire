from spellchecker import SpellChecker
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import streamlit as st

output = ""

def scrapText():
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    rows = soup.find_all('div', attrs={'class': 'ssrcss-1ocoo3l-Wrap e42f8511'})
    driver.close()

    obj = soup.find('img', attrs={'class': 'img-sized__img landscape'})
    if (obj):
        print(obj)

    title = ""
    ans = soup.find('h1', attrs={'class': 'ssrcss-15xko80-StyledHeading e10rt3ze0'})
    if (ans):
        title = ans.text

    text_arr = []
    for row in rows:
        sub_row = row.find_all('p', attrs={'class': 'ssrcss-1q0x1qg-Paragraph e1jhz7w10'})
        for sub_sub_row in sub_row:
            text_arr.append(sub_sub_row.text)

    text_arr.pop()
    article_text = ' '.join(text_arr)
    return title + " " + article_text

st.title('Article Recommendation')

scrapedText = "Here comes your article"
url = st.text_input("Article Link", value="")

if st.button("Enter"):
    scrapedText = scrapText()
    st.text(scrapedText)

from streamlit_card import card

card(
  title="Hello World!",
  text="Some description",
  image="https://png.pngtree.com/thumb_back/fh260/background/20210908/pngtree-test-papers-daytime-answer-sheet-classroom-exam-photography-map-with-map-image_822606.jpg",
  url="https://github.com/gamcoh/st-card"
)

card(
  title="Hello World 0!",
  text="Some description",
  image="https://png.pngtree.com/thumb_back/fh260/background/20210908/pngtree-test-papers-daytime-answer-sheet-classroom-exam-photography-map-with-map-image_822606.jpg",
  url="https://github.com/gamcoh/st-card"
)

card(
  title="Hello World 1!",
  text="Some description",
  image="https://png.pngtree.com/thumb_back/fh260/background/20210908/pngtree-test-papers-daytime-answer-sheet-classroom-exam-photography-map-with-map-image_822606.jpg",
  url="https://github.com/gamcoh/st-card"
)