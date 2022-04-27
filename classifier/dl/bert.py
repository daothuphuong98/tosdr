from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np
from classifier.classifier import Classifier

class BERTClassifier(Classifier):

    _name = 'bert'
    model_path = 'classifier/model/bert'
    output_dir = 'classifier/model/bert/res'

    def __init__(self):
        super().__init__()
        self.trainer = None
        self.data_collator = None
        self.tokenizer = None
        self.model = None

    @staticmethod
    def compute_metrics(eval_preds):
        metric = load_metric('accuracy')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def train(self, filepath):
        checkpoint = 'bert-base-uncased'
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        dataset = load_dataset('csv', data_files=filepath)
        dataset = dataset["train"].train_test_split(train_size=0.7, seed=42)
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

    def predict(self, filepath):
        dataset = load_dataset('csv', data_files=filepath)
        dataset = dataset.map(lambda x: self.tokenizer(x['sent'], truncation=True, padding='max_length'),
                              batched=True)
        dataset = dataset.remove_columns(['sent'])
        predictions = self.trainer.predict(dataset['train'])
        predictions = np.argmax(predictions.predictions, axis=-1)

        clf.score(dataset['train']['labels'], predictions)

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


if __name__=='__main__':
    clf = BERTClassifier()
    clf.load_model()
    clf.predict('data/tosware_test.csv')



