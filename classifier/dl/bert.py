from classifier.dl.dl_classifier import DLClassifier

class BERTClassifier(DLClassifier):

    _name = 'BERT'
    model_path = 'classifier/model/bert'
    output_dir = 'classifier/model/bert/res'
    checkpoint = 'bert-base-uncased'


if __name__=='__main__':
    clf = BERTClassifier(False)
    # clf.load_model()
    clf.predict('data/tosware_test.csv', 'data/prediction/test/BERT_nsw.csv')



