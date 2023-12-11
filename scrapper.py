from spellchecker import SpellChecker
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class Scrapper:
    def __init__(self) -> None:
        pass
        # self.options = Options()
        # self.options.headless = True
        # self.driver = webdriver.Chrome(options=self.options)

    def scrape_article(self, url):
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
        return [title , article_text]
    
    