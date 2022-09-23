import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv('../data/data.csv')
data_X = data.iloc[:,2:]
data_y = data.click.values
data_X = data_X.apply(LabelEncoder().fit_transform)
fields = data_X.max().values + 1
tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)

train_X = torch.from_numpy(train_X.values).long()
val_X = torch.from_numpy(val_X.values).long()
test_X = torch.from_numpy(test_X.values).long()

train_y = torch.from_numpy(train_y).long()
val_y = torch.from_numpy(val_y).long()
test_y = torch.from_numpy(test_y).long()

train_set = Data.TensorDataset(train_X, train_y)
val_set = Data.TensorDataset(val_X, val_y)
train_loader = Data.DataLoader(dataset=train_set,batch_size=128,shuffle=True)
val_loader = Data.DataLoader(dataset=val_set,batch_size=128,shuffle=False)

print('starting')

epoches = 5
def train(model):
    for epoch in range(epoches):
        train_loss = []
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            # print(y)
            # print(F.one_hot(y,7))
            loss = criterion(pred, F.one_hot(y,7).float().detach())
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        prediction = []
        y_true = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                loss = criterion(pred, F.one_hot(y,7).float().detach())
                val_loss.append(loss.item())
                pred = torch.argmax(pred,1)
                prediction.extend(pred.cpu().tolist())
                y_true.extend(y.tolist())
                # print(pred.cpu().tolist(),y_true)
        val_auc = roc_auc_score(y_true=y_true, y_score=prediction)
        print ("EPOCH %s train loss : %.5f   validation loss : %.5f   validation auc is %.5f" % (epoch, np.mean(train_loss), np.mean(val_loss), val_auc))        
    return train_loss, val_loss, val_auc

from model import AutoInt
model = AutoInt.AutoIntNet(feature_fields=fields, embed_dim=8, head_num = 2, 
                           attn_layers=3, mlp_dims=(32, 16), dropout=0.2).cuda()
_ = train(model)