import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)


class Classifier:
    """
    Interface class
    """
    _name = None

    def get_name(self):
        return self._name

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info('Apply model %s' % self.get_name())
        self.results = []

    def score(self, y_true, y_pred, y_pred_proba=None):
        self.log.info(classification_report(y_true, y_pred))
        self.log.info(confusion_matrix(y_true, y_pred))
        if y_pred_proba:
            self.log.info(f'AUC: {roc_auc_score(y_true, y_pred_proba)}')

    def train(self, filepath):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError