"""
The script carries the Gridsearch cross validation process for the Naive Bayes classifier.
It searches for the combination of the hyperparameters below in terms of best model evaluation score (accuracy):
* alpha - additive smoothing parameter
* fit_prior - whether to learn class prior probabilities or not.
GridsearchCV helps also to find out whether the using of the stop words or bigrams improve the model evaluation score.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


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

    # Create the pipeline object for the Naive Bayes classifier for multinomial models
    # with the use of the TfidfVectorizer and TfidfTransformer object
    mnb_pipeline = Pipeline([
        # Token pattern parameter is added in order to add the percent values to the features vector
        ('vect', TfidfVectorizer(token_pattern='\w\w+|[1-9]\.[1-9]\%|[1-9]\,[1-9]\%|[1-9]\.[1-9]|[1-9]\,[1-9]|[1-9]\%')),
        # The TfidfTransformer with default parameteres values
        ('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)),
        ('clfMNB', MultinomialNB()),
    ])
    # Fit Naive Bayes model according to train data
    mnb_pipeline.fit(X, y)

    # Find the best model parameters optimized by cross-validated grid-search over a parameter grid presented below
    grid_params = {
        'clfMNB__alpha': (np.linspace(0.5, 1.5, 6)),
        'clfMNB__fit_prior': (True, False),
        'vect__max_df': (np.linspace(0.1, 1, 10)),
        'vect__sublinear_tf': (True, False),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__stop_words': (stop_words_list, None),
    }

    # Find the best model parameters taking into account evaluation measure(s) inclued in the scorings list
    scorings = ['accuracy']
    labels = np.unique(df['kategoria'])
    cv = 5
    best_params = []
    for score in scorings:
        gs = GridSearchCV(mnb_pipeline, grid_params, cv=cv, n_jobs=-1, verbose=1, error_score=0, scoring=scorings,
                          refit=score)
        gs = gs.fit(X, y)
        best_params.append(gs.best_params_)
    df_best_params = pd.DataFrame(best_params)
    cv_results = df_best_params.to_excel('GridsearchCV_results_NB.xlsx')


if __name__ == '__main__':
    main()
