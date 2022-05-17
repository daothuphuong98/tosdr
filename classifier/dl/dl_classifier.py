from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, Dataset
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
import json
from gensim.parsing.preprocessing import remove_stopwords

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)

class DLClassifier():

    _name = None
    model_path = None
    output_dir = None
    optimal_threshold = None

    def __init__(self, stop_word = True):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info('Apply model %s' % self.get_name())

        self.trainer = None
        self.data_collator = None
        self.tokenizer = None
        self.model = None
        self.stopword = stop_word


        with open('data/cutoff.json') as f:
            cutoff = json.load(f)

        if self.stopword:
            self.model_path += '_sw'
            log_path = f'log/{self._name}_sw.log'
            self.optimal_threshold = cutoff.get(self._name + '_sw', {}).get('threshold')
        else:
            self.model_path += '_nsw'
            log_path = f'log/{self._name}_nsw.log'
            self.optimal_threshold = cutoff.get(self._name + '_nsw', {}).get('threshold')

        logging.basicConfig(filename=log_path,
                            filemode='a',
                            datefmt='%H:%M:%S',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            level=logging.INFO,
                            force=True)

    def get_name(self):
        return self._name

    def score(self, y_true, y_pred, y_pred_proba=None):
        self.log.info(classification_report(y_true, y_pred))
        self.log.info(confusion_matrix(y_true, y_pred))
        if y_pred_proba is not None:
            self.log.info(f'AUC: {roc_auc_score(y_true, y_pred_proba)}')

    @staticmethod
    def compute_metrics(eval_preds):
        metric = load_metric('f1')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def train(self, filepath):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if self.stopword:
            df = pd.read_csv(filepath)
            df['sent'] = df['sent'].apply(remove_stopwords)
            dataset = Dataset.from_pandas(df)
        else:
            dataset = load_dataset('csv', data_files=filepath)["train"]

        dataset = dataset.train_test_split(train_size=0.7, seed=42)
        dataset['validation'] = dataset.pop('test')
        dataset = dataset.map(lambda x: self.tokenizer(x['sent'], truncation=True),
                              batched=True)
        dataset = dataset.remove_columns(['sent'])

        batch_size = 8
        num_epochs = 5
        arg = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy='epoch',
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01
        )

        self.trainer = Trainer(
            model=self.model,
            args=arg,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def predict(self, filepath, proba=False):
        scoreboard = pd.DataFrame()
        if not proba:
            if self.stopword:
                df = pd.read_csv(filepath)
                df['sent'] = df['sent'].apply(remove_stopwords)
                dataset = Dataset.from_pandas(df)
            else:
                dataset = load_dataset('csv', data_files=filepath)['train']
            scoreboard['labels'] = dataset['labels']
            dataset = dataset.map(lambda x: self.tokenizer(x['sent'], truncation=True, padding='max_length'),
                                  batched=True)
            dataset = dataset.remove_columns(['sent'])
            predictions = self.trainer.predict(dataset).predictions

            proba_pred = softmax(predictions, axis=-1)[:, -1]
        else:
            df = pd.read_csv(proba)
            proba_pred = df['pred_proba']
            scoreboard['labels'] = df['labels']
        pred = np.where(proba_pred > self.optimal_threshold, 1, 0)
        scoreboard['predictions'] = pred
        scoreboard['pred_proba'] = proba_pred
        if self.stopword:
            scoreboard.to_csv(f'data/prediction/test/{self._name}_sw.csv', index=False)
        else:
            scoreboard.to_csv(f'data/prediction/test/{self._name}_nsw.csv', index=False)

        self.score(scoreboard['labels'], pred, proba_pred)

    def predict_sentence(self, sentence, proba = False, threshold=None):
        if self.stopword:
            sentence = remove_stopwords(sentence)
        dataset = Dataset.from_dict({'sent': [sentence]})
        dataset = dataset.map(lambda x: self.tokenizer(x['sent'], truncation=True, padding='max_length'),
                              batched=True)
        dataset = dataset.remove_columns(['sent'])
        predictions = self.trainer.predict(dataset).predictions

        proba_pred = softmax(predictions, axis=-1)
        if proba:
            return proba_pred[0,-1]
        else:
            if threshold is None:
                return int(proba_pred[0,-1] > self.optimal_threshold)
            else:
                return int(proba_pred[0,-1] > threshold)

    def save_model(self):
        if self.trainer:
            self.trainer.save_model(self.model_path)
        else:
            self.log.error('Trained model is required before saving')

    def load_model(self):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.trainer = Trainer(model=self.model)
        except:
            self.log.info('No model found')




