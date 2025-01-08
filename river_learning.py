import pathlib
import numpy as np
from collections import OrderedDict
from typing import Optional
import util.tool
import os
from util.online_record_reader import Record_reader
from river import preprocessing
from river import tree
from river import forest
from river import metrics
from river import linear_model
from river import naive_bayes
from river import ensemble
import time

def try_models(model, dataset):
    accu = metrics.Accuracy()
    F1 = metrics.F1()
    y_preds = []
    ys = []
    for x,y in dataset:
        y_pred = model.predict_one(x)
        model.learn_one(x,y)
        accu.update(y,y_pred)
        F1.update(y,y_pred)
        y_preds.append(y_pred)
        ys.append(y)
    print(type(model))
    print(accu)
    print(F1)
    subdirectory = 'results'  # 子目录名称
    filename = f"{type(model).__name__}.txt"  # 使用模型类型作为文件名

    # 创建子目录（如果不存在）
    os.makedirs(subdirectory, exist_ok=True)
    with open(os.path.join(subdirectory, filename), 'w') as f:
        for true, pred in zip(ys, y_preds):
            f.write(f"{true},{pred}\n")  # 每行写入 true 和 pred，用逗号分隔
    return

if __name__ == '__main__':
    reader = Record_reader("all.trace","./DynamoRIO-Linux-10.93.19965",hot_thred=0.0006899560670691673,training=True, eq_capacity=30)
    read_begin_time = time.process_time()
    reader.read()
    read_end_time = time.process_time()
    print(f"read time:{read_end_time - read_begin_time}")
    dataset = reader.train_set
    HoeffdingTree = tree.HoeffdingTreeClassifier()
    Hoeffding_Adaptive_Tree = tree.HoeffdingAdaptiveTreeClassifier()
    # linear_reg = linear_model.LinearRegression(intercept_lr=.1)
    NB = naive_bayes.GaussianNB()
    ARF = forest.ARFClassifier(seed=8, leaf_prediction="mc")
    ALMA = linear_model.ALMAClassifier()
    BAG = ensemble.BaggingClassifier(
        model=(
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression()
        ),
        n_models=3,
        seed=42
    )
    BAG_Tree = ensemble.BaggingClassifier(
        model= tree.HoeffdingTreeClassifier(
            split_criterion='gini',
            delta = 1e-5,
            grace_period=2000
        ),
        n_models=5,
        seed=42
    )
    ALMA_preprocess = (
        preprocessing.StandardScaler() |
        linear_model.ALMAClassifier()
    )
    ada_model = ensemble.AdaBoostClassifier(
        model=(
            tree.HoeffdingTreeClassifier(
                split_criterion='gini',
                delta=1e-5,
                grace_period=2000
            )
        ),
        n_models=5,
        seed=42
    )
    NB_begin_time = time.process_time()
    try_models(NB,dataset)
    NB_end_time = time.process_time()
    print(f"NB time:{NB_end_time - NB_begin_time}")

    tree_begin_time = time.process_time()
    try_models(HoeffdingTree,dataset)
    tree_end_time = time.process_time()
    print(f"Hoeffdingtree time:{tree_end_time - tree_begin_time}")

    hoeffding_adaptive_tree_begin_time = time.process_time()
    try_models(Hoeffding_Adaptive_Tree,dataset)
    hoeffding_adaptive_tree_end_time = time.process_time()
    print(f"Hoeffding adaptive tree time:{hoeffding_adaptive_tree_end_time - hoeffding_adaptive_tree_begin_time}")

    # linear_reg_begin_time = time.process_time()
    # try_models(linear_reg,dataset)
    # linear_reg_end_time = time.process_time()
    # print(f"linear_reg time:{linear_reg_end_time - linear_reg_begin_time}")
    
    AdaBoost_begin_time = time.process_time()
    try_models(ada_model,dataset)
    AdaBoost_end_time = time.process_time()
    print(f"AdaBoost time:{AdaBoost_end_time-AdaBoost_begin_time}")

    ARF_begin_time = time.process_time()
    try_models(ARF,dataset)
    ARF_end_time = time.process_time()
    print(f"ARF time:{ARF_end_time-ARF_begin_time}")

    ALMA_begin_time = time.process_time()
    try_models(ALMA,dataset)
    ALMA_end_time = time.process_time()
    print(f"ALMA time:{ALMA_end_time - ALMA_begin_time}")

    BAG_begin_time = time.process_time()
    try_models(BAG,dataset)
    BAG_end_time = time.process_time()
    print(f"bagging time:{BAG_end_time - BAG_begin_time}")

    BAG_Tree_time_begin = time.process_time()
    try_models(BAG_Tree,dataset)
    BAG_Tree_time_end = time.process_time()
    print(f"bagging time:{BAG_Tree_time_end - BAG_Tree_time_begin}")

    try_models(ALMA_preprocess,dataset)