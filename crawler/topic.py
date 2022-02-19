from crawler import BasicCrawler
from selenium.webdriver.common.by import By

class TopicCrawler(BasicCrawler):

    URL = 'https://edit.tosdr.org/topics'

    def __init__(self, db):
        super().__init__(db)

    def crawl(self):
        self.driver.get(self.URL)

        results = []
        count = 0
        topics = self.driver.find_elements(By.CSS_SELECTOR, '.table.table-striped > tbody > tr')
        for tp in topics:
            name = tp.find_element(By.CSS_SELECTOR, 'td:nth-child(2)').text
            link = tp.find_element(By.CSS_SELECTOR, 'td:nth-child(2) > a').get_attribute('href')
            ids = link.split('/')[-1]
            link = self.URL+'/'+ids
            results.append({'TOPIC_ID': ids,
                            'NAME': name,
                            'LINK': link})
            count += 1

            if len(results) > 30:
                self.db.insert('TOPIC', results)
                self.log.info(f'Crawl {count} results')
                results = []

        self.db.insert('TOPIC', results)
        self.log.info(f'Crawl {count} results')
        self.log.info(f'Finished')

if __name__ == '__main__':
    crawler = TopicCrawler('raw')
    crawler.crawl()




