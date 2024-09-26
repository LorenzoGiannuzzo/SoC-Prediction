import pandas as pd
import numpy as np
from df_importer import HPPC_Dis, HPPC_Cha, OCV_data
import matplotlib.pyplot as plt

OCV_Cha = OCV_data.iloc[:,[1,2]]
OCV_Dis = OCV_data.iloc[:,[4,5]]

HPPC_Dis = HPPC_Dis.iloc[:,[0,1,2,3]]
HPPC_Cha = HPPC_Cha.iloc[:,[0,1,2,3]]

