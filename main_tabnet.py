# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:12:46 2021

@author: tiantian05
"""
# import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from pytorch_tabnet.tab_model import TabNetRegressor

#导入构造好的数据
data = pd.read_csv('F:/restaurant2020/train_firm_mean.csv')
# =============================================================================
# 新加的生成数据
# =============================================================================
#generate_data = pd.read_csv('F:/restaurant2020/tabnet2021/generate_population.csv')
#generate_data.columns = data.columns
#data = pd.concat([data,generate_data],axis=0,ignore_index=True)


test = pd.read_csv('F:/restaurant2020/test_firm_mean.csv')
#训练数据和标签
y = data['firm']
train_data = data.drop(['firm'],axis=1)
#测试数据和标签
test_y = test['firm']
test_data = test.drop(['firm'],axis=1)

#构建模型
clf = TabNetRegressor(
        n_d = 32,
        n_a = 32,
        n_steps = 4,
        gamma = 1.3,
        n_shared = 1,
        n_independent = 2,
        seed = 2020,
        lambda_sparse = 1e-3,
        cat_dims=[], cat_emb_dim=[], cat_idxs=[])
#损失函数
cetr = torch.nn.L1Loss()
#验证集的预测值数组
oof_tab = np.zeros(len(train_data))
#测试集的预测值
prediction_tab = np.zeros(len(test_data))

trn_data, val_data, trn_y, val_y = train_test_split(train_data, y, test_size=0.2, random_state=42)
trn_data = trn_data.values
trn_y = trn_y.values.reshape(-1,1)
val_data = val_data.values
val_y = val_y.values.reshape(-1,1)
clf.fit(trn_data,trn_y,
        eval_set = [(val_data,val_y)],
        eval_metric = ['mae'],
        max_epochs = 150,
        loss_fn = cetr,
        patience = 50,
        batch_size = 256, 
        virtual_batch_size = 128,
        num_workers = 0,
        drop_last = False
        )
clf.f.close()

prediction_y = clf.predict(test_data.values)
#employment
#r2_1000=r2_score(test_y[0:317],prediction_y[0:317])
#r2_1500=r2_score(test_y[317:530],prediction_y[317:530])
#r2_2000=r2_score(test_y[530:686],prediction_y[530:686])
#r2_2500=r2_score(test_y[686:807],prediction_y[686:807])
#r2_3000=r2_score(test_y[807:1195],prediction_y[807:1195])
#r2_3500=r2_score(test_y[1195:1275],prediction_y[1195:1275])
#r2_4000=r2_score(test_y[1275:1343],prediction_y[1275:1343])
#r2_4500=r2_score(test_y[1343:1400],prediction_y[1343:1400])
#r2_5000=r2_score(test_y[1400:],prediction_y[1400:])


##consumption
#r2_1000=r2_score(test_y[0:256],prediction_y[0:256])
#r2_1500=r2_score(test_y[256:437],prediction_y[256:437])
#r2_2000=r2_score(test_y[437:574],prediction_y[437:574])
#r2_2500=r2_score(test_y[574:683],prediction_y[574:683])
#r2_3000=r2_score(test_y[683:771],prediction_y[683:771])
#r2_3500=r2_score(test_y[771:844],prediction_y[771:844])
#r2_4000=r2_score(test_y[844:906],prediction_y[844:906])
#r2_4500=r2_score(test_y[906:959],prediction_y[906:959])
#r2_5000=r2_score(test_y[959:],prediction_y[959:])

##firm
r2_1000=r2_score(test_y[0:305],prediction_y[0:305])
r2_1500=r2_score(test_y[305:514],prediction_y[305:514])
r2_2000=r2_score(test_y[514:667],prediction_y[514:667])
r2_2500=r2_score(test_y[667:787],prediction_y[667:787])
r2_3000=r2_score(test_y[787:885],prediction_y[787:885])
r2_3500=r2_score(test_y[885:965],prediction_y[885:965])
r2_4000=r2_score(test_y[965:1032],prediction_y[965:1032])
r2_4500=r2_score(test_y[1032:1089],prediction_y[1032:1089])
r2_5000=r2_score(test_y[1089:],prediction_y[1089:])


#population
#r2_1000=r2_score(test_y[0:315],prediction_y[0:315])
#r2_1500=r2_score(test_y[315:527],prediction_y[315:527])
#r2_2000=r2_score(test_y[527:682],prediction_y[527:682])
#r2_2500=r2_score(test_y[682:803],prediction_y[682:803])
#r2_3000=r2_score(test_y[803:1192],prediction_y[803:1192])
#r2_3500=r2_score(test_y[1192:1272],prediction_y[1192:1272])
#r2_4000=r2_score(test_y[1272:1339],prediction_y[1272:1339])
#r2_4500=r2_score(test_y[1339:1396],prediction_y[1339:1396])
#r2_5000=r2_score(test_y[1396:],prediction_y[1396:])

print('网格id为1000的拟合系数为{:.4f}'.format(r2_1000))
print('网格id为1500的拟合系数为{:.4f}'.format(r2_1500))
print('网格id为2000的拟合系数为{:.4f}'.format(r2_2000))
print('网格id为2500的拟合系数为{:.4f}'.format(r2_2500))
print('网格id为3000的拟合系数为{:.4f}'.format(r2_3000))
print('网格id为3500的拟合系数为{:.4f}'.format(r2_3500))
print('网格id为4000的拟合系数为{:.4f}'.format(r2_4000))
print('网格id为4500的拟合系数为{:.4f}'.format(r2_4500))
print('网格id为5000的拟合系数为{:.4f}'.format(r2_5000))
