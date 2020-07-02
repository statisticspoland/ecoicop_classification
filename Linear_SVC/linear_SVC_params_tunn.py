"""
The script carries the Gridsearch cross validation process for the Linear SVC classifier.
It searches for the combination of the hyperparameters below in terms of best model evaluation score (accuracy):
* penalty - norm used in the penalization
* multi_class - determines the multi-class strategy
* C - the strength of the regularization is inversely proportional to C
* loss - specifies the loss function.
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def main():
    df = pd.read_excel("..\products_allshops_dataset.xlsx", names=['produkt', 'kategoria'])
    np.random.seed(5)
    df = df.reindex(np.random.permutation(df.index))

    df_train = df[3000:].copy()
    df_valid = df[1500:3000].copy()
    df_test = df[0:1500].copy()
    X = df_train['produkt']
    y = df_train['kategoria']

    # Getting the list of polish stop words
    stop_words_file = '..\polish_stopwords.txt'
    with open(stop_words_file, mode='r') as stop_words:
        stop_words_list = stop_words.read().split('\n')

    """
    Create the pipeline object for the Linear SVC classifier
    with the use of the TfidfVectorizer and TfidfTransformer object.
    Due to software limitations the vectorizer parameters are already given.
    Token pattern parameter is added in order to add the percent values to the features vector.
    """
    vect_pipeline = Pipeline([
        ('vect', TfidfVectorizer(max_df=0.1, ngram_range=(1, 2), stop_words=stop_words_list, sublinear_tf=True, \
                                 token_pattern='\w\w+|[1-9]\.[1-9]\%|[1-9]\,[1-9]\%|[1-9]\.[1-9]|[1-9]\,[1-9]|[1-9]\%')),
        ('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)),
    ])

    linSVC_pipeline = Pipeline([
        ('vect_pipe', vect_pipeline),
        # The value of the dual parameter below is suggested in case of n_samples > then n_features.
        ('clfLin', svm.LinearSVC(dual=False, max_iter=1200)),
    ])
    # Fit Naive Bayes model according to train data
    linSVC_pipeline.fit(X, y)
    # Find the best model parameters optimized by cross-validated grid-search over a parameter grid presented below
    grid_params = {
        'clfLin__penalty': ('l1', 'l2'),
        'clfLin__multi_class': ('ovr', 'crammer_singer'),
        'clfLin__C': (0.01, 0.1, 1, 10, 100, 1000),
        'clfLin__loss': ('hinge', 'squared_hinge'),
    }
    # Find the best model parameters taking into account evaluation measure(s) inclued in the scorings list
    scorings = ['accuracy']
    labels = np.unique(df['kategoria'])
    cv = 5
    best_params = []
    for score in scorings:
        gs = GridSearchCV(linSVC_pipeline, grid_params, cv=cv, n_jobs=-1, verbose=1, error_score=0, scoring=scorings,
                          refit=score)
        gs = gs.fit(X, y)
        best_params.append(gs.best_params_)
    df_best_params = pd.DataFrame(best_params)
    cv_results = df_best_params.to_excel('GridsearchCV_results_LinSVC.xlsx')


if __name__ == '__main__':
    main()
