import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font",size=14)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
sns.set(style = "white")
sns.set(style = "whitegrid", color_codes = True)


data = pd.read_csv('banking.csv',header = 0)
data = data.dropna()

cols = ['age','duration','pdays','emp_var_rate','cons_price_idx']
X = data[cols]
y = data['y']

import statsmodels.api as sm
logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())