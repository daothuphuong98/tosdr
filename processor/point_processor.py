import pandas as pd
import requests
import bs4

df = pd.read_csv('clean_point.csv')
services = [182, 194, 536, 217, 230, 225, 1553, 219, 198, 193, 175, 200, 297, 2428, 280, 1815, 244, 180, 2453, 462]
to_crawl = df[(df['STATUS'] == 'DECLINED') & (df['COMMENT']).isnull()]

count = 0
for ind, row in to_crawl.iterrows():
    try:
        req = requests.get(row['LINK'])
        soup = bs4.BeautifulSoup(req.text, 'lxml')
        # text = soup.select_one("blockquote")
        # if text:
        #     df.loc[ind, 'TEXT'] = ' '.join(text.text.splitlines()[:-2]).strip()
        comments = soup.select('.card-inline-service > p')
        comments = [c.text for c in comments]
        comments = '\n '.join([x for x in comments if len(x) < 150])
        df.loc[ind, 'COMMENT'] = comments
        if count % 50 == 0:
            print(count)
        count += 1
    except:
        print(row['LINK'])

df.to_csv('clean_point2.csv', index=False)

