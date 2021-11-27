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
train = pd.read_csv('./data/population_auto.csv')   #
target = 'population'
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
# =============================================================================
# 构建模型
# =============================================================================
clf = TabNetRegressor(
    n_d=128, n_a = 128, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=[],cat_dims=[],cat_emb_dim=[],
    lambda_sparse=1e-4, momentum=0.2, clip_value=2.,
    optimizer_fn= torch.optim.Adam,
    optimizer_params=dict(lr=0.005),
    scheduler_params = {"gamma": 0.5,
                        "milestones":[10,21,30]},
    scheduler_fn = torch.optim.lr_scheduler.MultiStepLR, epsilon=1e-15
)

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_name=['valid'],
    eval_metric=['mae'],
    max_epochs = 150, patience=100,
    batch_size=512, virtual_batch_size=256
)

preds = clf.predict(X_test)
y_true = y_test
test_score = mean_absolute_error(y_pred=preds, y_true=y_true)

print(f"BEST VALID SCORE : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR : {test_score}")