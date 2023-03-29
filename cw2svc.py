# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:30:37 2022

@author: lboyd
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


from sklearn.svm import SVC
clf = SVC(kernel='rbf', degree=3, C=1.0)
clf.fit(training_instances, training_labels)
predicted_test_labels = clf.predict(test_instances)



from sklearn.metrics import classification_report 
print(classification_report(test_labels, predicted_test_labels, digits=3))
print(accuracy_score(test_labels, predicted_test_labels))