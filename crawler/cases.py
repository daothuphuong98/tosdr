from crawler import BasicCrawler
from selenium.webdriver.common.by import By
import bs4
import requests

class CaseCrawler(BasicCrawler):

    URL = 'https://edit.tosdr.org/cases'

    def __init__(self, db):
        super().__init__(db)

    def crawl(self):
        self.driver.get(self.URL)

        results = []
        count = 0
        cases = self.driver.find_elements(By.CSS_SELECTOR, '.table.table-striped > tbody > tr')
        for case in cases:
            link = case.find_element(By.CSS_SELECTOR, 'td > a').get_attribute('href')
            topic_id = case.find_element(By.CSS_SELECTOR, 'td:nth-child(2) > a').get_attribute('href')
            topic_id = topic_id.split('/')[-1]
            result = self.crawl_case(link, topic_id)
            results.append(result)
            count += 1

            if len(results) > 30:
                self.db.insert('CASES', results)
                self.log.info(f'Crawl {count} results')
                results = []

        self.db.insert('CASES', results)
        self.log.info(f'Crawl {count} results')
        self.log.info(f'Finished')

    def crawl_case(self, link, topic_id):
        req = requests.get(link)
        ids = link.split('/')[-1]
        soup = bs4.BeautifulSoup(req.text, 'lxml')
        card = soup.select_one('div.card-inline div.row')
        title = card.select_one('.col-lg-6 > h3').get_text()
        desc = card.select_one('.col-lg-6 > p').get_text()
        rating = card.select_one('.col-lg-6.text-right > p').get_text()
        try:
            rating = rating.split(':')[-1].strip()
        except:
            rating = None
        return {'CASE_ID': ids,
                'LINK': link,
                'TOPIC_ID': topic_id,
                'TITLE': title,
                'DESCRIPTION': desc,
                'RATING': rating}


if __name__ == '__main__':
    crawler = CaseCrawler('raw')
    crawler.crawl()




