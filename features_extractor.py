import pandas as pd
import numpy as np

from df_explorer import HPPC_Dis, HPPC_Cha, OCV_Cha, OCV_Dis

train_data = pd.concat([HPPC_Dis, HPPC_Cha.iloc[3:,[0,1,2,3]]], axis=0)

train_data.iloc[:, 1] = train_data.iloc[:, 1].fillna(train_data.iloc[:, 4])
time  = np.arange(0, len(train_data.iloc[:, 0])-2)
train_data.iloc[3:,0] = time

