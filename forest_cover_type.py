from pytorch_tabnet.tab_model import  TabNetClassifier
import torch
import numpy as np
np.random.seed(0)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
target = "Covertype"

bool_columns = [
    "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
    "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
    "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
    "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
    "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
    "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
    "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
    "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
    "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
    "Soil_Type40"
]

int_columns = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

feature_columns = (
        int_columns + bool_columns + [target])

#import data
train = pd.read_csv('./data/forest-cover-type.csv',header = None,names = feature_columns)
#split train,val and test
if 'set' not in train.columns:
    train['set'] = np.random.choice(['train','valid','test'],p=[0.8,0.1,0.1],size = (train.shape[0],))
train_indices = train[train.set =='train'].index
valid_indices = train[train.set == 'valid'].index
test_indices = train[train.set == 'test' ].index

categorical_columns = [ "Aspect", "Slope","Hillshade_3pm",
    "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
    "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
    "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
    "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
    "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
    "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
    "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
    "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
    "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
    "Soil_Type40"
]

unused = ['set']
features = [col for col in train.columns if col not in unused+[target]]
cat_idxs = [i for i,f in enumerate(features) if f in categorical_columns]
cat_dims = [train[col].nunique() for col in categorical_columns]
#cat_emb_dim = [200,200,200]+[2]*44

#network parameters
clf = TabNetClassifier(
    n_d = 64,n_a=64,n_steps = 5,
    gamma = 0.5,n_independent=2,n_shared=2,
    cat_idxs = cat_idxs,
    cat_dims = cat_dims,cat_emb_dim=1,
    lambda_sparse=1e-4,momentum=0.3,clip_value=2,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=0.01),
    scheduler_params = {"gamma": 0.9,
                        "milestones":[5,10,20]},
    scheduler_fn = torch.optim.lr_scheduler.MultiStepLR, epsilon=1e-15
)
X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]

clf.fit(
    X_train=X_train,y_train = y_train,
    eval_set=[(X_train,y_train),(X_valid,y_valid)],
    eval_name=['train','valid'],
    max_epochs=400,patience=100,
    batch_size=1024,virtual_batch_size=512
)

#predict
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_pred=y_pred,y_true=y_test)
print(f"FINAL TEST SCORE:{test_acc}")



