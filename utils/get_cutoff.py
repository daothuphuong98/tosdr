import pandas as pd
from os import listdir
import numpy as np
from sklearn.metrics import roc_curve, f1_score
import json

def get_cutoff(folder, n_sample=4):
    files = listdir(folder)
    result = {}
    for file in files:
        filename = file.split('.csv')[0]
        orig_df = pd.read_csv(folder + file)
        max_score = []
        for i in range(n_sample):
            df = orig_df.sample(frac=0.2, random_state=i*10)
            tpr, tpr, thresholds = roc_curve(df['labels'], df['pred_proba'])
            accuracy_ls = []
            for thres in thresholds:
                y_pred = np.where(df['pred_proba'] >= thres, 1, 0)
                accuracy_ls.append({'threshold': thres, 'score': f1_score(df['labels'], y_pred)})
            max_score.append(max(accuracy_ls, key=lambda x: x['score']))
        max_score = pd.DataFrame(max_score)
        result[filename] = {'threshold': max_score['threshold'].mean(), 'score': max_score['score'].mean()}
    with open('data/cutoff.json', 'a') as f:
        json.dump(result, f)

if __name__ == '__main__':
    get_cutoff('data/prediction/train/')