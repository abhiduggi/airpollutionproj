import numpy as np
import pandas as pd

df=pd.read_csv("no2train.csv")
df=df.values
df2=df[:,2:6]
df3 = (df2 - df2.mean(axis=0)) / df2.astype(int).std(axis=0)

df7=pd.read_csv("no2test.csv")
df7=df7.values
df8=df7[:,2:6]
df9 = (df8 - df8.mean(axis=0)) / df8.astype(int).std(axis=0)

def next_batch_train(window_size, loc, batch_size, n_inputs, n_outputs):
    x=np.reshape(df3[loc-1:window_size+loc-1,:],(batch_size,window_size,n_inputs))
    y=np.reshape(df3[loc:window_size+loc,:],(batch_size,window_size,n_outputs))         
    return [x,y]


def next_batch_test(window_size, loc, batch_size, n_inputs, n_outputs):
    x1=np.reshape(df9[loc-1:window_size+loc-1,:],(batch_size,window_size,n_inputs))
    y1=np.reshape(df9[loc:window_size+loc,:],(batch_size,window_size,n_outputs))         
    return [x1,y1]