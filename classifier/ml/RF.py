from classifier.ml.ml_classifier import MLClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier

class RFClassifier(MLClassifier):

    _name = 'Random Forest'
    model_path = 'classifier/model/rf'

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

        self.md = RandomForestClassifier(bootstrap=False, max_features='sqrt',
                                    min_samples_split=5, n_estimators=180, random_state=40)
        self.md.fit(smoted_X, smoted_y)

if __name__ == '__main__':
    clf = RFClassifier(True)
    # clf.train('data/tosware_train.csv')
    # clf.save_transformer()
    clf.load_model()
    clf.predict('data/tosware_test.csv')




