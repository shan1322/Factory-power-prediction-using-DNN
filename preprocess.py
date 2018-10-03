import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
df=pd.read_excel('Data/Folds5x2_pp.xlsx')
AT,V,AP,RH,PE=np.array(df['AT']),np.array(df['V']),np.array(df['AP']),np.array(df['RH']),np.array(df['PE'])
AT,V,AP,RH,PE=preprocessing.scale(AT),preprocessing.scale(V),preprocessing.scale(AP),preprocessing.scale(RH),preprocessing.scale(PE)
np.save('mat/AT.npy',AT)
np.save('mat/V.npy',V)
np.save('mat/AP.npy',AP)
np.save('mat/RH.npy',RH)
np.save('mat/PE.npy',PE)