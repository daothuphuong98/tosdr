import logging

from app.utils import Methods
from statistics import mode, mean
from classifier.ml.LGBM import LtGBMClassifier
from classifier.ml.RF import RFClassifier
from classifier.ml.SVC import SVClassifier
from classifier.dl.bert import BERTClassifier
from classifier.dl.roberta import ROBERTAClassifier
import pandas as pd
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay

from nltk.tokenize.punkt import PunktSentenceTokenizer as pt

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)


class TOSService:

    start_mark =  '<span style="background-color: #FFFF00"> '
    end_mark =  '</span> '
    prediction_folder = 'data/prediction/test'

    def __init__(self):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.log.setLevel(logging.INFO)

        self.tokenizer = pt()

        self.classifiers_sw = {
            Methods.RF: RFClassifier(True),
            Methods.SVC: SVClassifier(True),
            Methods.LGBM: LtGBMClassifier(True),
            Methods.BERT: BERTClassifier(True),
            Methods.ROBERTA: ROBERTAClassifier(True)
        }

        self.classifiers_nsw = {
            Methods.RF: RFClassifier(False),
            Methods.SVC: SVClassifier(False),
            Methods.LGBM: LtGBMClassifier(False),
            Methods.BERT: BERTClassifier(False),
            Methods.ROBERTA: ROBERTAClassifier(False)
        }

        for name, c in self.classifiers_sw.items():
            self.log.info('Load model %s with stop words' % name)
            c.load_model()

        for name, c in self.classifiers_nsw.items():
            self.log.info('Load model %s with no stop words' % name)
            c.load_model()

    def detect(self, s, methods, stopword, threshold, voting='hard'):
        if voting == 'hard':
            proba = False
        else:
            proba = True

        labels = []
        for method in methods:
            if stopword:
                classifier = self.classifiers_sw.get(method)
            else:
                classifier = self.classifiers_nsw.get(method)
            if classifier is None:
                raise Exception('Classifier %s not found' % method)
            labels.append(classifier.predict_sentence(s, proba, threshold))

        if proba:
            label = int(mean(labels) >= 0.5)
        else:
            label = mode(labels)

        return {'method': methods, 'sentence': s, 'label': label}

    def detect_paragraph(self, paragraph, methods, stopword, threshold, voting='hard'):
        sents = self.tokenizer.tokenize(text=paragraph)

        for ind, s in enumerate(sents):
            result = self.detect(s, methods, stopword, threshold, voting)
            if result['label'] == 1:
                sents[ind] = self.start_mark + s + self.end_mark

        return {'method': methods,
                'paragraph': ' '.join(sents)}

    def score_test_dataset(self, methods, stopword, voting='hard'):
        score = pd.DataFrame()
        for method in methods:
            if stopword:
                pred_path = self.prediction_folder + f"/{method}_sw.csv"
            else:
                pred_path = self.prediction_folder + f"/{method}_nsw.csv"
            method_pred = pd.read_csv(pred_path)

            if voting == 'hard':
                score[method] = method_pred['predictions']
            else:
                score[method] = method_pred['pred_proba']

        proba_pred = score.mean(axis=1)
        if voting == 'soft':
            score['prediction'] = pd.to_numeric(proba_pred >= 0.5)
        else:
            score['prediction'] = score.mode(axis=1).loc[:,0]
        score['proba_pred'] = proba_pred

        report = classification_report(method_pred['labels'], score['prediction'])
        roc_curve = RocCurveDisplay.from_predictions(method_pred['labels'], score['proba_pred'])
        conf_matrix = ConfusionMatrixDisplay.from_predictions(method_pred['labels'], score['prediction'])

        return report, roc_curve.figure_, conf_matrix.figure_

if __name__ == '__main__':
    x = TOSService()
    x.detect_paragraph('''PLEASE READ THESE TERMS OF USE CAREFULLY. 
    BY REGISTERING FOR AN ACCOUNT OR BY ACCESSING OR USING THE SERVICES, YOU AGREE TO BE BOUND BY THESE TERMS OF USE AND ALL TERMS INCORPORATED BY REFERENCE.''',
                       ["Light GBM"], True)