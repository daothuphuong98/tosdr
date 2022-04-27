from classifier.classifier import Classifier
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

class EnsembleClassifier(Classifier):

    _name = 'Ensemble'
    model_path = 'classifier/model/es'
    transformer_path = 'classifier/model/tfidf'

    def __init__(self):
        super().__init__()

    def train(self, filepath):
        df=pd.read_csv(filepath)

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectorizer.fit(df['sent'])
        X = self.vectorizer.transform(df['sent'])
        y = df['Labels']

        os = SMOTEENN(sampling_strategy=0.5, random_state=40)
        self.log.info(f'Number of positive labels: {(y==1).sum()}')
        self.log.info(f'Number of negative labels: {len(y)-(y==1).sum()}')
        self.log.info(f'Start over-sampling using SMOTEENN')

        smoted_X, smoted_y = os.fit_resample(X, y)
        self.log.info(f'After over-sampling, number of positive labels: {(smoted_y==1).sum()}')
        self.log.info(f'After over-sampling, number of negative labels: {len(smoted_y) - (smoted_y==1).sum()}')

        self.md = VotingClassifier([('svc', SVC(random_state=40,probability=True)),
                                    ('rf', RandomForestClassifier(bootstrap=False,
                                                                  max_features='sqrt',
                                                                  min_samples_split=5,
                                                                  n_estimators=180,
                                                                  random_state=40)),
                                    # ('gb', GradientBoostingClassifier(random_state=40))
                                    ],
                      voting='soft')
        self.md.fit(smoted_X, smoted_y)

    def predict(self, filepath):
        df = pd.read_csv(filepath)

        X = self.vectorizer.transform(df['sent'])
        y = df['labels']

        pred = self.md.predict(X)
        pred_proba = self.md.predict_proba(X)[:,-1]

        self.score(y, pred, pred_proba)

        df['predictions'] = pred
        df[(y == 1) & (pred != 1)].to_csv(f'data/prediction/{self._name}_fn.csv', index=False)
        df[(y != 1) & (pred == 1)].to_csv(f'data/prediction/{self._name}_fp.csv', index=False)
        df.to_csv(f'data/prediction/{self._name}.csv', index=False)

    def predict_sentence(self, sentence):
        X = self.vectorizer.transform([sentence])
        pred = self.md.predict(X)
        return pred

    def save_model(self):
        if self.md:
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
        if self.vectorizer:
            joblib.dump(self.vectorizer, self.transformer_path)
        else:
            self.log.error('Trained model is required before saving')

    def load_transformer(self):
        try:
            self.vectorizer = joblib.load(self.transformer_path)
        except:
            self.log.info('No model found')

if __name__ == '__main__':
    clf = EnsembleClassifier()
    clf.train('data/tosware_train.csv')
    clf.save_model()
    clf.predict('data/tosware_test.csv')




