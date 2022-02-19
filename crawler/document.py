import pandas as pd
from crawler import BasicCrawler
import bs4
import requests

class DocumentCrawler(BasicCrawler):

    URL = 'https://edit.tosdr.org/documents'
    CRAWL_NUM = 3

    def __init__(self, db, webdriver_init = False):
        super().__init__(db, webdriver_init)

    def crawl(self):
        results = []

        while True:
            docs = pd.read_csv('resources/document.csv', sep='\t')
            to_crawl = docs[docs['crawled'].isnull()].iloc[:self.CRAWL_NUM, :]
            if len(to_crawl) < 1:
                break

            for ind, doc in to_crawl.iterrows():
                result = self.crawl_case(doc['link'])
                results.append(result)

            self.db.insert('DOCUMENT', results)
            docs.loc[to_crawl.index, 'crawled'] = 1
            docs.to_csv('resources/document.csv', sep='\t', index=False)
            self.log.info(f'Crawl {len(results)} results, finished {docs["crawled"].sum()}/{len(docs)}')
            results = []

        self.log.info(f'Finished')

    def crawl_case(self, link):
        req = requests.get(link)
        ids = link.split('/')[-1]
        soup = bs4.BeautifulSoup(req.text, 'lxml')
        card = soup.select_one('div.card-inline div.row')
        service_id = card.select_one('.col-lg-6 > h1 > a').attrs['href']
        service_id = service_id.split('/')[-2]
        type = card.select_one('.col-lg-6 > h5').get_text()
        text = soup.select_one('.overflow').get_text()
        return {'DOC_ID': ids,
                'SERVICE_ID': service_id,
                'TEXT': text,
                'TYPE': type,
                'LINK': link}

if __name__ == '__main__':
    crawler = DocumentCrawler('raw')
    crawler.crawl()




