import logging

from api.utils import Methods
from classifier.ml.ES import EnsembleClassifier
from classifier.ml.GB import GBClassifier
from classifier.ml.RF import RFClassifier
from classifier.ml.SVC import SVClassifier

from nltk.tokenize.punkt import PunktSentenceTokenizer as pt

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)


class TOSService:

    start_mark =  '\033[91m'
    end_mark =  '\033[0m'

    def __init__(self):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.log.setLevel(logging.INFO)

        self.classifiers = {
            Methods.RF: RFClassifier(),
            Methods.SVC: SVClassifier(),
            Methods.GB: GBClassifier(),
            Methods.ES: EnsembleClassifier()
        }
        for name, c in self.classifiers.items():
            self.log.info('Load model %s' % name)
            c.load_model()

    def detect(self, sentence, method):
        classifier = self.classifiers.get(method)
        if classifier is None:
            return None
        label = classifier.predict(sentence)
        res = {
            "sentence": sentence,
            "label": label,
            "method": method
        }
        return res

    def detect_paragraph(self, paragraph, method):
        classifier = self.classifiers.get(method)
        if classifier is None:
            return None

        sents = pt.span_tokenize(paragraph)
        results =[]

        for s in sents:
            sentence = paragraph[s[0]:s[1]]
            label = classifier.predict(sentence)
            res = {
                "sentence": sentence,
                "position": s,
                "method": method,
                'label': label
            }
            results.append(res)
        return results
