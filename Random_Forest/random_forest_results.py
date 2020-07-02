import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, matthews_corrcoef

df = pd.read_excel("../products_allshops_dataset.xlsx", names=['produkt', 'kategoria'])
labels = np.unique(df['kategoria'])

# It is possible to choose different random seed (and the results will be different for each random seed)
np.random.seed(102)
df = df.reindex(np.random.permutation(df.index))

# Dividing the data into the training, validation and testing group
df_test = df[0:1500].copy()
df_validation = df[1500:3000].copy()
df_training = df[3000:].copy()

# Create an instance of the CountVectorizer object
vectorizer = CountVectorizer(token_pattern='\w\w+|[1-9]\.[1-9]\%|[1-9]\,[1-9]\%|[1-9]\.[1-9]|[1-9]\,[1-9]|[1-9]\%')
# Map each word in our training narratives to a vector position
vectorizer.fit(df_training['produkt'])
# Convert each training narrative to its vector representation and stack them into a matrix
x_training = vectorizer.transform(df_training['produkt'])
y_training = df_training['kategoria']

# Classifier parameters that were giving the best results
clf = RandomForestClassifier(n_estimators=200,
                             criterion='gini',
                             min_samples_leaf=1,
                             min_samples_split=3,
                             max_features='log2',
                             bootstrap=False,
                             oob_score=False,
                             warm_start=False,
                             class_weight=None)
# We fit the model to our training data (calculate the model parameters)
clf.fit(x_training, y_training)
# Generate predicted codes for our training narratives
y_training_predicted = clf.predict(x_training)
accuracy_training = accuracy_score(y_training, y_training_predicted)
classification_report_training = classification_report(y_training, y_training_predicted)

"""
We already counted the validation accuracy when we were choosing the best parameters.
Now we would like to present all the results for both the validation and test set. 
"""

x_validation = vectorizer.transform(df_validation['produkt'])
y_validation_predicted = clf.predict(x_validation)
y_validation = df_validation['kategoria']
accuracy_validation = accuracy_score(y_validation, y_validation_predicted)
classification_report_validation = classification_report(y_validation, y_validation_predicted)

x_test = vectorizer.transform(df_test['produkt'])
y_test_predicted = clf.predict(x_test)
y_test = df_test['kategoria']
accuracy_test = accuracy_score(y_test, y_test_predicted)
classification_report_test = classification_report(y_test, y_test_predicted)

# You can count the micro F1 score for training, validation and testing group
f1_score_micro_training = f1_score(y_training, y_training_predicted, average='micro')
f1_score_micro_validation = f1_score(y_validation, y_validation_predicted, average='micro')
f1_score_micro_test = f1_score(y_test, y_test_predicted, average='micro')

# You can count the Matthews correlation coefficient for training, validation and testing group
mcc_training = matthews_corrcoef(y_training, y_training_predicted)
mcc_validation = matthews_corrcoef(y_validation, y_validation_predicted)
mcc_test = matthews_corrcoef(y_test, y_test_predicted)

"""
Printing all the data on the screen:
- classification reports
- accuracy
- F1-score
- MCC
"""
print("Classifiction reports:")
print("TRAINING DATA")
print(classification_report_training)
print("VALIDATION DATA")
print(classification_report_validation)
print("TEST DATA")
print(classification_report_test)
print("=" * 30)
print("TRAINING DATA")
print("accuracy = {}".format(accuracy_training))
print("F1-score = {}".format(f1_score_micro_training))
print("MCC = {}".format(mcc_training))
print("=" * 30)
print("VALIDATON DATA")
print("accuracy = {}".format(accuracy_validation))
print("F1-score = {}".format(f1_score_micro_validation))
print("MCC = {}".format(mcc_validation))
print("=" * 30)
print("TEST DATA")
print("accuracy = {}".format(accuracy_test))
print("F1-score = {}".format(f1_score_micro_test))
print("MCC = {}".format(mcc_test))
print("=" * 30)

"""
The commented code below can be used to export the classifier and vectorizer to joblib file
which can be loaded into other files or applications. 
"""
# # Export the classifier and vectorizer to the file that can be used in an application to classify products
# from joblib import dump
# dump(clf, 'random_forest.joblib')
# dump(vectorizer, 'vectorizer_rf.joblib')

"""
The data frame with test data can be saved into the xlsx file.
It is also possible to add new columns - for example the most probable category and its probability.
"""
df_test['Autocode'] = y_test_predicted
y_pred_prob = clf.predict_proba(x_test)
df_test['Probability'] = y_pred_prob.max(axis=1)
df_test.to_excel("rf_autocode.xlsx")

"""
It is also possible to create confusion matrix for each data set
The best way to show the confusion matrix is by using Python library called Seaborn.
It is possible to save the generated chart into the separate file.
"""

cm_training = confusion_matrix(y_training, y_training_predicted)
cm_validation = confusion_matrix(y_validation, y_validation_predicted)
cm_test = confusion_matrix(y_test, y_test_predicted)

# Create the heatmap with confusion matrix (validation) using Seaborn library
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm_validation, cmap='Greens', annot=True, fmt="d", ax=ax, xticklabels=labels, yticklabels=labels). \
    set_title("Confusion matrix - validation")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Create the heatmap with confusion matrix (test) using Seaborn library
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm_test, cmap='Greens', annot=True, fmt="d", ax=ax, xticklabels=labels, yticklabels=labels). \
    set_title("Confusion matrix - test")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
