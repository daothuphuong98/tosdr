import logging
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)


class Classifier:
    """
    Interface class
    """
    _name = None
    transformer_path = 'classifier/model/tfidf'
    model_path = None
    optimal_threshold = None

    def get_name(self):
        return self._name

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info('Apply model %s' % self.get_name())
        self.results = []
        self.vectorizer = None
        self.md = None

    def score(self, y_true, y_pred, y_pred_proba=None):
        self.log.info(classification_report(y_true, y_pred))
        self.log.info(confusion_matrix(y_true, y_pred))
        if y_pred_proba is not None:
            self.log.info(f'AUC: {roc_auc_score(y_true, y_pred_proba)}')

    def train(self, filepath):
        raise NotImplementedError

    def predict(self, filepath):
        df = pd.read_csv(filepath)

        X = self.vectorizer.transform(df['sent'])
        y = df['labels']

        pred_proba = self.md.predict_proba(X)[:,-1]
        pred = np.where(pred_proba > self.optimal_threshold, 1, 0)

        self.score(y, pred, pred_proba)

        scoreboard = pd.DataFrame()
        scoreboard['predictions'] = pred
        scoreboard['pred_proba'] = pred_proba
        scoreboard['labels'] = y
        if self.stopword:
            scoreboard.to_csv(f'data/prediction/{self._name}_sw.csv', index=False)
        else:
            scoreboard.to_csv(f'data/prediction/{self._name}_nsw.csv', index=False)

    def predict_sentence(self, sentence, proba=False, threshold=None):
        X = self.vectorizer.transform([sentence])
        proba_pred = self.md.predict_proba(X)[0,-1]
        if proba:
            return proba_pred
        else:
            if threshold is None:
                return int(proba_pred > self.optimal_threshold)
            else:
                return int(proba_pred > threshold)

    def save_model(self):
        if self.md is not None:
            joblib.dump(self.md, self.model_path)
        else:
            self.log.error('Trained model is required before saving')

    def load_model(self):
        try:
            self.load_transformer()
            self.md = joblib.load(self.model_path)
        except:
            self.log.info('No model found')

    def save_transformer(self):
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, self.transformer_path)
        else:
            self.log.error('Trained model is required before saving')

    def load_transformer(self):
        try:
            self.vectorizer = joblib.load(self.transformer_path)
        except:
            self.log.info('No model found')