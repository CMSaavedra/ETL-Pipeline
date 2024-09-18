import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization

# Load the first 5 million rows of the training dataset
train_df =  pd.read_csv('../input/train.csv', nrows = 5_000_000)

# Add travel vector features to the dataframe
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)

# Remove data with missing values
train_df = train_df.dropna(how = 'any', axis = 'rows')

# Remove data with erroneous values
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]

print(train_df.dtypes)
print('New size: %d' % len(train_df))

print(train_df.head())
