import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os

train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000) 
train_df.dtypes #check the data types of the columns

