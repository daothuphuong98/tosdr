from html_processor import HTMLProcessor
from lang_processor import LanguageProcessor
import pandas as pd
from utils.sqldb import MySQLDB

def main(ind):
    # Hoang: 0-5
    # Ngoc Anh: 6-13
    # Phuong: 13-19
    # docs = pd.read_csv('data/doc_count.tsv', sep = '\t')
    # service = docs.loc[ind, 'SERVICE_ID']
    db = MySQLDB('34.131.19.243', 'user', 'tosdr2022', 'raw')
    docs = db.query(f'SELECT * FROM DOCUMENT WHERE DOC_ID = 317')
    procs = [HTMLProcessor(), LanguageProcessor()]
    for ind, doc in docs.iterrows():
        for proc in procs:
            doc['TEXT'] = proc.process(doc['TEXT'])
        if doc['TEXT'] is not None:
            doc.to_csv(f'data/clean_data/clean_doc/doc_{doc["DOC_ID"]}.csv', index=False)

if __name__ == '__main__':
    main(20)

