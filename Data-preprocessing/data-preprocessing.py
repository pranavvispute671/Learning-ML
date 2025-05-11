import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv("data.csv")

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

print(data.isnull().sum())

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  #strategy can be mean, median, most_frequent, constant
imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# OneHot Encoding- Converts categorical to individual binary columns

ct = ColumnTransformer(transformers=[('encoder','OneHotEncoder',[0])], remainder='passthrough')
# Here the transformer is a list of tuples
# transformers = [('typeoftransformer','nameoftransformer',[indeofcolumns])]
# if remainders is not given, only transformed column will be kept
x = np.array(ct.fit_transform(x)) # need to typecast, since fit_transform doesnt return a numpy array
print(x)

