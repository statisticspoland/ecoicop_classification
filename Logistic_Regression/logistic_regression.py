"""
The code is counting the best accuracy ad F1 score for Logistic Regression Model and different values of C.
The code is testing also other parameters: fit_intercept, class_weight, solver and multi_class.
The regularization hyperparamether C control how closely we allow the model to fit the training data.
If C is too high the model will tend to overfit. The optimal value for C can be found by experimentation.
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_excel("../products_allshops_dataset.xlsx", names=['produkt', 'kategoria'])

np.random.seed(5)
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
testing different parameters for logistic regression model:
* c
* fit_intercept
* class_weight
* solver
* multi_class
"""
for c in [0.1, 1, 2, 3]:
    for fit_intercept in [True, False]:
        for class_weight in [None, 'balanced']:
            for solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]:
                for multi_class in ["ovr", "multinomial"]:
                    max_iter = 200
                    print("C={}, fit_intercept={}, class_weight={}, solver={}, multi_class={}".
                          format(c, fit_intercept, class_weight, solver, multi_class))
                    try:
                        clf = LogisticRegression(C=c, fit_intercept=fit_intercept, class_weight=class_weight,
                                                 solver=solver,
                                                 multi_class=multi_class, max_iter=max_iter)
                        # we fit the model to our training data (ie. we calculate the model parameters)
                        clf.fit(x_training, y_training)
                        # Convert the validation narratives to a feature matrix
                        x_validation = vectorizer.transform(df_validation['produkt'])
                        # Generate predicted codes for our validation narratives
                        y_validation_pred = clf.predict(x_validation)
                        # Calculate how accurately these match the true codes
                        y_validation = df_validation['kategoria']
                        val_accuracy = accuracy_score(y_validation, y_validation_pred)
                        # you can also check the accuracy on the training data
                        y_training_pred = clf.predict(x_training)
                        train_accuracy = accuracy_score(y_training, y_training_pred)
                        f1_score_micro = f1_score(y_validation, y_validation_pred, average='micro')
                        list_of_results.append(
                            [c, fit_intercept, class_weight, solver, multi_class, max_iter, val_accuracy,
                             train_accuracy, f1_score_micro])
                    except ValueError:
                        print("impossible to solve")

df_to_excel = pd.DataFrame(list_of_results)
df_to_excel.to_excel("results_logistic_regression.xlsx")
