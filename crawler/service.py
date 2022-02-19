from crawler import BasicCrawler
from selenium.webdriver.common.by import By

class ServiceCrawler(BasicCrawler):

    URL = 'https://edit.tosdr.org/services'

    def __init__(self, db):
        super().__init__(db)

    def crawl(self):
        login_stat = self.login()
        if not login_stat:
            raise Exception('Failed to login')

        self.driver.get(self.URL)

        results = []
        count = 0
        services = self.driver.find_elements(By.CSS_SELECTOR, '#myTableBody > tr')
        for sv in services:
            name = sv.find_element(By.CSS_SELECTOR, 'td:nth-child(2)').text
            if 'deprecated' in name.lower():
                continue
            rating = sv.find_element(By.CSS_SELECTOR, 'td:nth-child(3)').text
            link = sv.find_element(By.CSS_SELECTOR, '.text-right > a').get_attribute('href')
            ids = link.split('/')[-1]
            link = self.URL+'/'+ids
            results.append({'SERVICE_ID': ids,
                            'NAME': name,
                            'LINK': link,
                            'RATING': rating})
            count += 1

            if len(results) > 30:
                self.db.insert('SERVICE', results)
                self.log.info(f'Crawl {count} results')
                results = []

        self.db.insert('SERVICE', results)
        self.log.info(f'Crawl {count} results')
        self.log.info(f'Finished')

if __name__ == '__main__':
    crawler = ServiceCrawler('raw')
    crawler.crawl()




