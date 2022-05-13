import logging

from api.utils import Methods
from statistics import mode, mean
from classifier.ml.LGBM import GBClassifier
from classifier.ml.RF import RFClassifier
from classifier.ml.SVC import SVClassifier

from nltk.tokenize.punkt import PunktSentenceTokenizer as pt

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)


class TOSService:

    start_mark =  '<span style="background-color: #FFFF00"> '
    end_mark =  '</span> '

    def __init__(self):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.log.setLevel(logging.INFO)

        self.tokenizer = pt()
        self.classifiers = {
            Methods.RF: RFClassifier(),
            Methods.SVC: SVClassifier(),
            Methods.GB: GBClassifier()
        }
        for name, c in self.classifiers.items():
            self.log.info('Load model %s' % name)
            c.load_model()

    def detect_paragraph(self, paragraph, methods, voting='hard'):
        if voting == 'hard':
            proba = False
        else:
            proba = True

        sents = self.tokenizer.tokenize(text=paragraph)

        for ind, s in enumerate(sents):
            labels = []
            for method in methods:
                classifier = self.classifiers.get(method)
                if classifier is None:
                    raise Exception('Classifier %s not found' % method)
                if proba:
                    labels.append(classifier.predict_sentence(s, proba)[0,-1])
                else:
                    labels.append(classifier.predict_sentence(s, proba)[0])

            if proba:
                label = int(mean(labels) >= 0.5)
            else:
                label = mode(labels)

            if label == 1:
                sents[ind] = self.start_mark + s + self.end_mark

        return {'method': methods,
                'paragraph': ' '.join(sents)}

