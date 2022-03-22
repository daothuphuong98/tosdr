from processor import Processor
import pandas as pd
from nltk import tokenize
from html_processor import HTMLProcessor
from thefuzz.fuzz import partial_ratio
from os import listdir
from tqdm import tqdm

class SentenceProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.html_proc = HTMLProcessor()

    def sentence_split(self, path):
        self.error = []
        try:
            entry = pd.read_csv(path, sep='\n', names =['doc'])
            sent_path = path.replace('doc', 'sent')
            entry.index = ['index','doc_id', 'service_id', 'text', 'type','link']

            entry.loc['text', 'doc'] = entry.loc['text', 'doc'].replace('\n', ' ').replace('\r', '')
            sentences = pd.Series(tokenize.sent_tokenize(entry.loc['text', 'doc']))
            word_num = sentences.apply(lambda x: len(x.split()))
            sentences = sentences[word_num >= 5]
            sentences.to_csv(sent_path, sep='\n', index=False, header=False)
        except Exception as e:
            print("'"+path+"'")

    def match_score(self, sent, point_text):
        if len(point_text) > len(sent):
            return partial_ratio(sent, point_text)
        else:
            return partial_ratio(point_text, sent)

    def point_match(self, point, sents):
        point_text = self.html_proc.process(point['TEXT'])
        if len(point_text) < 3:
            return
        match_score = sents.apply(self.match_score, args=(point_text, ))
        match_result = pd.concat({'match_score':match_score, 'sent':sents},
                                 axis=1).sort_values('match_score', ascending=False)
        if (match_result['match_score'] == 100).sum() > 0:
            for ind, row in match_result[match_result['match_score'] == 100].iterrows():
                self.match.loc[ind, 'point_id'].append(point['POINT_ID'])
                self.match.loc[ind, 'point_text'].append(point_text)
                self.match.loc[ind, 'match_score'].append(100)
                if (match_result['match_score'] == 100).sum() > 1:
                    self.match.loc[ind, 'flag'] += f'Many sentences match for point {point["POINT_ID"]}\n'
        else:
            for ind, row in match_result[match_result['match_score'] > 0].head(1).iterrows():
                self.match.loc[ind, 'point_id'].append(point['POINT_ID'])
                self.match.loc[ind, 'point_text'].append(point_text)
                self.match.loc[ind, 'match_score'].append(row['match_score'])
                self.match.loc[ind, 'point_title'] += point['TITLE']
                if row['match_score'] < 90:
                    self.match.loc[ind, 'flag'] += f'Low match score for point {point["POINT_ID"]}\n'

    def doc_match(self, doc_path, point_path):
        doc = pd.read_csv(doc_path, encoding= 'utf-8', sep='\t', header=None, engine='python', quoting=3)
        doc = doc.iloc[:, 0]
        doc_id= doc_path.split('/')[-1][5:-4]

        point = pd.read_csv(point_path)
        point = point[point['DOC_ID'] == int(doc_id)]

        self.match = pd.DataFrame()
        self.match['sent'] = doc
        self.match['point_id'] = [[] for _ in range(len(self.match))]
        self.match['point_text'] = [[] for _ in range(len(self.match))]
        self.match['match_score'] = [[] for _ in range(len(self.match))]
        self.match['point_title'] = ''
        self.match['flag'] = ''

        for ind, row in point.iterrows():
            self.point_match(row, doc)
        self.match['point_id'] = self.match['point_id'].apply(lambda x: ','.join([str(i) for i in x]))
        self.match['point_text'] = self.match['point_text'].apply(lambda x: '\n'.join(x))
        self.match['match_score'] = self.match['match_score'].apply(lambda x: ','.join([str(i) for i in x]))
        self.match.to_csv(f'data/match_sent/csv/match_{doc_id}.csv', sep='|', encoding='utf-8')

x = SentenceProcessor()
err= []
dir_path ='data/clean_data/clean_sent/'
for p in tqdm(listdir(dir_path)):
    try:
        x.doc_match(dir_path+p, 'data/clean_data/clean_point.csv')
    except Exception as e:
        print(p, e)
        err.append(dir_path+p)
print(err)





            













