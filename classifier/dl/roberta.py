from classifier.dl.dl_classifier import DLClassifier

class ROBERTAClassifier(DLClassifier):

    _name = 'ROBERTA'
    model_path = 'classifier/model/roberta'
    output_dir = 'classifier/model/roberta/res'
    checkpoint = 'roberta-base'

if __name__=='__main__':
    clf = ROBERTAClassifier(True)
    # clf.load_model()
    clf.predict('data/tosware_test.csv', 'data/prediction/test/ROBERTA_sw.csv')



