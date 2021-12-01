# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:48:23 2021

@author: tiantian05
"""

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
np.random.seed(0)
import warnings
warnings.filterwarnings('ignore')


#导入构造好的数据
train = pd.read_csv('F:/restaurant2020/tabnet2021/data/house.csv')
target = 'Prices'
# =============================================================================
# 划分数据集
# =============================================================================
if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index

# =============================================================================
# 训练所使用的的features
# =============================================================================
unused_feat = ['Set']
features = [ col for col in train.columns if col not in unused_feat+[target]]

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]#.reshape(-1, 1)

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]#.reshape(-1, 1)

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]#.reshape(-1, 1)

min_res = float('inf')
#构建模型
for i in range(8,12):
#    for j in [0.01,0.5,0.1]:
    for k in [100,500,1000]:
            clf =  XGBRegressor(
            max_depth = i,
#            num_leaves = 2**i-1,
            #learning_rate = j,
            n_estimators = k,
            #num_round = 500,
            )



            clf.fit(X_train,y_train,
#                    eval_set = [(X_valid,y_valid)],
#                    eval_metric = ['mae'],
#                    early_stopping_rounds = 200,
#                    verbose = 100
                    )
            preds = clf.predict(X_valid)
            valid_mae = mean_absolute_error(y_pred=preds, y_true=y_valid)
            preds_test = clf.predict(X_test)
            test_mae = mean_absolute_error(y_pred = preds_test,y_true = y_test)
            min_res = min(min_res,test_mae)
            print(min_res)

