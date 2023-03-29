# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:39:12 2022

@author: 24514471
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker

import pandas as pd
def load_data_from_csv(input_csv):
    df = pd.read_csv(input_csv, header=0)
    csv_headings = list(df.columns.values)
    feature_names = csv_headings[:len(csv_headings) - 1]
    label_name = csv_headings[len(csv_headings) - 1:len(csv_headings)][0]
    df = df._get_numeric_data()
#    numpy_array = df.as_matrix() #deprecated
#    numpy_array = df.values #deprecated
    numpy_array = df.to_numpy()
    number_of_rows, number_of_columns = numpy_array.shape
    instances = numpy_array[:, 0:number_of_columns - 1]
    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    return feature_names, instances, labels


input_training_csv=(r"C:\Users\lboyd\OneDrive\Documents\reviews_Video_Games_training.csv")
input_test_csv=(r"C:\Users\lboyd\OneDrive\Documents\reviews_Video_Games_test.csv")
training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv=input_training_csv)
test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv=input_test_csv)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=20)
classifier.fit(training_instances, training_labels)
predicted_test_labels = classifier.predict(test_instances)


acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(training_instances, training_labels)
    predicted_test_labels = neigh.predict(test_instances)
    acc.append(metrics.accuracy_score(test_labels, predicted_test_labels))
    
plt.figure(figsize=(20,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.draw()
plt.show()

from sklearn.metrics import classification_report 
print(classification_report(test_labels, predicted_test_labels, digits=3))
print(accuracy_score(test_labels, predicted_test_labels))