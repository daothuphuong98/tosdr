import pandas as pd
from crawler import BasicCrawler
import bs4
import requests

class PointCrawler(BasicCrawler):

    URL = 'https://edit.tosdr.org'
    CRAWL_NUM = 30

    def __init__(self, db, webdriver_init = False):
        super().__init__(db, webdriver_init)

    def crawl(self):
        results = []
        errors = []

        while True:
            docs = pd.read_csv('resources/point.csv', sep='\t')
            to_crawl = docs[docs['crawled'].isnull()].iloc[:self.CRAWL_NUM, :]
            if len(to_crawl) < 1:
                break

            for ind, doc in to_crawl.iterrows():
                try:
                    result = self.crawl_point(doc['link'])
                    results.append(result)
                except:
                    errors.append(doc['link'])

            self.db.insert('POINT', results)
            docs.loc[to_crawl.index, 'crawled'] = 1
            docs.to_csv('resources/point.csv', sep='\t', index=False)
            self.log.info(f'Crawl {len(results)} results, finished {docs["crawled"].sum()}/{len(docs)}')
            results = []

        self.log.info(f'Finished')

    def crawl_point(self, link):
        req = requests.get(link)
        ids = link.split('/')[-1]
        soup = bs4.BeautifulSoup(req.text, 'lxml')
        card = soup.select_one('.container > div:nth-of-type(2)')

        service_id = card.select_one('div:nth-of-type(1) > a').attrs['href']
        service_id = service_id.split('/')[-1]
        status = card.select_one('div:nth-of-type(2) > .label').get_text().strip()
        case_id = card.select_one('div:nth-of-type(3) > a').attrs['href']
        case_id = case_id.split('/')[-1]
        source = card.select_one('div:nth-of-type(5) > a')
        if source is not None:
            source = source.attrs['href']
        text = soup.select_one('.container blockquote')
        doc_id = None
        if text is not None:
            doc_id = text.select_one('cite > a').attrs['href']
            doc_id = doc_id.split('doc_')[-1]
            text = text.find_all(text=True, recursive=False)[0]
        else:
            text = soup.select_one('.container > .row > .col-sm-10')
            if text is not None:
                text = text.get_text()

        return {'POINT_ID': ids,
                'DOC_ID': doc_id,
                'SERVICE_ID': service_id,
                'CASE_ID': case_id,
                'STATUS': status,
                'SOURCE': source,
                'TEXT': text,
                'LINK': link}

    def get_links(self):
        cases_link = self.db.query('select link from CASES')
        res = []
        for ind, link in enumerate(cases_link['link']):
            req = requests.get(link)
            soup = bs4.BeautifulSoup(req.text, 'lxml')
            point_links = soup.select('#myTableBody > tr > td:nth-of-type(1) > a')
            point_links = [self.URL+x.attrs['href'] for x in point_links]
            res += point_links

            if ind % 10 == 9:
                self.log.info(f'Crawled {ind + 1} items')

        pd.Series(res).to_csv('resources/point.csv', sep='\t', index=False)

if __name__ == '__main__':
    crawler = PointCrawler('raw')
    crawler.crawl()




