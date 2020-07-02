import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, matthews_corrcoef, f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

"""
Implementation of the GridsearchCV results in order to generate
the most important model evaluation metrics for the validation and test dataset:
- classification report
- accuracy
- F1-score
- MCC
- confusion matrix
"""


def draw_cmatrix(y_set, predicted, labels, title):
    """
    Function that generates and shows the confusion matrix where parameters below are given:
    * y_set - categories (labels) that describe the X dataset - test or validation
    * predicted - matrix of the predicted categories
    * labels - all unique categories
    * title - part of the chart title
    """
    cm = confusion_matrix(y_set, predicted, labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, cmap='Greens', annot=True, fmt='d', ax=ax,
                linewidths=.5, xticklabels=labels, yticklabels=labels) \
        .set_title("Confusion matrix - " + title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def main():
    df = pd.read_excel("..\products_allshops_dataset.xlsx", names=['produkt', 'kategoria'])
    np.random.seed(5)
    df = df.reindex(np.random.permutation(df.index))
    labels = np.unique(df['kategoria'])

    df_train = df[3000:].copy()
    df_valid = df[1500:3000].copy()
    df_test = df[0:1500].copy()
    X = df_train['produkt']
    y = df_train['kategoria']
    X_valid = df_valid['produkt']
    y_valid = df_valid['kategoria']
    X_test = df_test['produkt']
    y_test = df_test['kategoria']

    stop_words_file = '..\polish_stopwords.txt'
    # Getting the list of polish stop words
    with open(stop_words_file, mode='r') as stop_words:
        stop_words_list = stop_words.read().split('\n')

    # Pipeline for feature vectorizing with the GridsearchCV results included
    vect_pipeline = Pipeline([
        ('vect', TfidfVectorizer(max_df=0.1, ngram_range=(1, 2), stop_words=stop_words_list, sublinear_tf=True)),
        ('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)),
    ])
    # Pipeline for Naive Bayes model with the GridsearchCV results included
    mnb_pipeline = Pipeline([
        ('vectpip', vect_pipeline),
        ('clf', MultinomialNB(alpha=0.5, fit_prior=False)),
    ])

    # Fit Naive Bayes model according to train data
    mnb_pipeline.fit(X, y)

    # Get the predicted codes for training set, validation and test set
    predicted_train = mnb_pipeline.predict(X)
    predicted_test = mnb_pipeline.predict(X_test)
    predicted_valid = mnb_pipeline.predict(X_valid)

    # Compare train, test and validation accuracy in case of overfitting
    accuracy_train = accuracy_score(y, predicted_train)
    accuracy_test = accuracy_score(y_test, predicted_test)
    accuracy_valid = accuracy_score(y_valid, predicted_valid)

    # Generate measures for each class of the classification for the training, validation and testing group
    report_train = classification_report(y, predicted_train, labels=labels)
    report_test = classification_report(y_test, predicted_test, labels=labels)
    report_valid = classification_report(y_valid, predicted_valid, labels=labels)

    """
    Check f1 score for training, validation and testing group.
    When true positive + false positive == 0 or true positive + false negative == 0,
    f-score returns 0 and raises UndefinedMetricWarning.
    """
    f1_score_micro_train = f1_score(y, predicted_train, average='micro')
    f1_score_micro_test = f1_score(y_test, predicted_test, average='micro')
    f1_score_micro_valid = f1_score(y_valid, predicted_valid, average='micro')

    # Count the Matthews correlation coefficient for the training, validation and testing group
    mcc_train = matthews_corrcoef(y, predicted_train)
    mcc_test = matthews_corrcoef(y_test, predicted_test)
    mcc_valid = matthews_corrcoef(y_valid, predicted_valid)

    # Let's summarize and print all the results for the Naive Bayes classifier according to the dataset sort
    print("\nGeneral metrics for")
    print("TRAINING DATA")
    print("=" * 30)
    print("accuracy = {}".format(accuracy_train))
    print("F1-score = {}".format(f1_score_micro_train))
    print("MCC = {}\n".format(mcc_train))
    print("*" * 10, "CLASSIFICATION REPORT", "*" * 10)
    print(report_train)
    print("=" * 30)
    print("General metrics for")
    print("VALIDATION DATA")
    print("=" * 30)
    print("accuracy = {}".format(accuracy_valid))
    print("F1-score = {}".format(f1_score_micro_valid))
    print("MCC = {}\n".format(mcc_valid))
    print("*" * 10, "CLASSIFICATION REPORT", "*" * 10)
    print(report_valid)
    print("=" * 30)
    print("General metrics for")
    print("TEST DATA")
    print("=" * 30)
    print("accuracy = {}".format(accuracy_test))
    print("F1-score = {}".format(f1_score_micro_test))
    print("MCC = {}\n".format(mcc_test))
    print("*" * 10, "CLASSIFICATION REPORT", "*" * 10)
    print(report_test)

    """
    The commented code below can be used to export the fitted model into joblib file.
    Next it can be loaded to other files or applications.
    """
    # Export the fitted classifier to the file that can be used in applications to classify products
    # and get the probabilities of predicted categories
    # from joblib import dump
    # dump(mnb_pipeline, 'naive_bayes.joblib')
    """
    Test data and their predicted values can be saved into the xlsx file.
    It is also possible to add new columns - for example the most probable category and its probability.
    """
    df_test['Autocode'] = predicted_test
    predicted_prob = mnb_pipeline.predict_proba(X_test)
    df_test['Probability'] = predicted_prob.max(axis=1)
    df_test.to_excel("nb_autocode.xlsx")

    """
    It is also possible to create confusion matrix for each data set with the use of Seaborn library.
    It shows how accurate does the classifier predicts the labels versus labels in the initial dataset (test or validation).
    The generated chart can be saved into the separate file.
    """
    # Drawing confusion matrix for the test and validation group
    cm_test = draw_cmatrix(y_test, predicted_test, labels, "test")
    cm_valid = draw_cmatrix(y_valid, predicted_valid, labels, "validation")


if __name__ == '__main__':
    main()
