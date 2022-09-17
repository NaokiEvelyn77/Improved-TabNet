# =============================================================================
# 导入相应的包
# =============================================================================
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import os
import torch

#导入构造好的数据
train = pd.read_csv('./data/sarcos.csv')   #
target = 'label'
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
y_train = train[target].values[train_indices].reshape(-1, 1)

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices].reshape(-1, 1)

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices].reshape(-1, 1)
#print(X_valid.shape)
#print(X_test.shape)

#define new model and load save parametres
loaded_clf = TabNetRegressor()
loaded_clf.load_model('sacros_model_no_diff.zip')

explain_matrix,masks = loaded_clf.explain(X_test[:32,:])

np.save('explain_matrix.npy',masks)
res = []
for key,value in masks.items():
    temp = np.array(value)
    temp = np.int64(temp >= 0.7)
    res.append(temp.sum(axis = 0))
print(res)
print(len(res),len(res[0]))






