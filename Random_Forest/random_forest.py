"""
The code is counting the best accuracy ad F1 score for Random Forest Model.
Tested hyperparameters:
criterion, min_samples_leaf, min_samples_split, max_features, bootstrap, oob_score, warm_start, class_weight.
The results are saved into the Excel file and can be further analysed.
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_excel("../products_allshops_dataset.xlsx", names=['produkt', 'kategoria'])
labels = np.unique(df['kategoria'])

np.random.seed(105)
df = df.reindex(np.random.permutation(df.index))

df_test = df[0:1500].copy()
df_validation = df[1500:3000].copy()
df_training = df[3000:].copy()

# Create an instance of the CountVectorizer object
vectorizer = CountVectorizer()
# Map each word in our training narratives to a vector position
vectorizer.fit(df_training['produkt'])
# Convert each training narrative to its vector representation and stack them into a matrix
x_training = vectorizer.transform(df_training['produkt'])

y_training = df_training['kategoria']
list_of_results = []

"""
checking the best parameters for random forest model:
* criterion
* min_samples_leaf
* min_samples_split
* max_features
* bootstrap
* oob_score
* warm_start
* class_weight
the results are saved in the file results_random_forests.xlsx
"""
for n_estimators in [30, 100, 150, 200]:
    for criterion in ['gini', 'entropy']:
        for min_samples_leaf in [1, 2, 3]:
            for min_samples_split in [2, 3]:
                for max_features in [None, "sqrt", "log2"]:
                    for bootstrap in [True, False]:
                        for oob_score in [True, False]:
                            for warm_start in [True, False]:
                                for class_weight in [None, 'balanced', 'balanced_subsample']:
                                    try:

                                        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                                                     min_samples_leaf=min_samples_leaf,
                                                                     min_samples_split=min_samples_split,
                                                                     max_features=max_features, bootstrap=bootstrap,
                                                                     oob_score=oob_score, warm_start=warm_start,
                                                                     class_weight=class_weight)
                                        print(
                                            "n_estimators={}, criterion={}, min_samples_leaf={}, min_samples_split={}, max_features={}, bootstrap={},oob_score={}, warm_start={}, class_weight={}".
                                                format(n_estimators, criterion, min_samples_leaf, min_samples_split,
                                                       max_features, bootstrap, oob_score, warm_start, class_weight))
                                        # we fit the model to our training data (ie. we calculate the model parameters)
                                        clf.fit(x_training, y_training)
                                        # Convert the validation narratives to a feature matrix
                                        x_validation = vectorizer.transform(df_validation['produkt'])
                                        # Generate predicted codes for our validation narratives
                                        y_validation_pred = clf.predict(x_validation)
                                        # Calculate how accurately these match the true codes
                                        y_validation = df_validation['kategoria']
                                        val_accuracy = accuracy_score(y_validation, y_validation_pred)
                                        y_training_pred = clf.predict(x_training)
                                        train_accuracy = accuracy_score(y_training, y_training_pred)
                                        f1_score_micro = f1_score(y_validation, y_validation_pred, average='micro')
                                        list_of_results.append(
                                            [n_estimators, criterion, min_samples_leaf, min_samples_split,
                                             max_features, bootstrap, oob_score, warm_start, class_weight,
                                             val_accuracy, train_accuracy, f1_score_micro])
                                    except ValueError:
                                        print(ValueError)

df_to_excel = pd.DataFrame(list_of_results)
df_to_excel.to_excel("results_random_forests.xlsx")
