import pathlib
import numpy as np
from collections import OrderedDict
from typing import Optional
import structure.util
from structure.record_reader import Record_reader
from river import datasets
from river import tree
from river import metrics

if __name__ == '__main__':
    reader = Record_reader("500ktrace.log","/home/zsq/DynamoRIO-Linux-10.0.0/logs","trace")
    reader.read()
    dataset = reader.train_set
    tree = tree.HoeffdingTreeClassifier()
    accu = metrics.Accuracy()
    for x,y in dataset:
        y_pred = tree.predict_one(x)
        tree.learn_one(x,y)
        accu.update(y,y_pred)
    print(accu)



