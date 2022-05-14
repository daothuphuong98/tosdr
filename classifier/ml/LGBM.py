from classifier.classifier import Classifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier

class LtGBMClassifier(Classifier):

    _name = 'Light GBM'
    model_path = 'classifier/model/lgbm'
    optimal_threshold = 0.609

    def __init__(self, stop_word):
        self.stopword = stop_word
        if stop_word:
            self.model_path += '_sw'
            self.transformer_path += '_sw'
        else:
            self.model_path += '_nsw'
            self.transformer_path += '_nsw'

        super().__init__()

    def train(self, filepath):
        df=pd.read_csv(filepath)

        if self.stopword:
            self.vectorizer = TfidfVectorizer(stop_words='english')
        else:
            self.vectorizer = TfidfVectorizer()

        self.vectorizer.fit(df['sent'])
        X = self.vectorizer.transform(df['sent'])
        y = df['labels']

        os = SMOTEENN(sampling_strategy=0.5, random_state=40)
        self.log.info(f'Number of positive labels: {(y==1).sum()}')
        self.log.info(f'Number of negative labels: {len(y)-(y==1).sum()}')
        self.log.info(f'Start over-sampling using SMOTEENN')

        smoted_X, smoted_y = os.fit_resample(X, y)
        self.log.info(f'After over-sampling, number of positive labels: {(smoted_y==1).sum()}')
        self.log.info(f'After over-sampling, number of negative labels: {len(smoted_y) - (smoted_y==1).sum()}')

        param = {'colsample_bytree': 0.5, 'learning_rate': 0.2, 'max_depth': 12, 'min_child_weight': 1,
                 'subsample': 0.9124868783261229}
        self.md = LGBMClassifier(random_state=40, **param)
        self.md.fit(smoted_X, smoted_y)

if __name__ == '__main__':
    clf = LtGBMClassifier(True)
    # clf.train('data/tosware_train.csv')
    # clf.save_transformer()
    clf.load_model()
    clf.predict('data/tosware_test.csv')




