import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv('mydata.csv')
data.head(10)
data.shape
data.count()

# 1st commit start (data description added)
print("="*40)
print()
print("Descriptive Statistics:")
print(data.describe())
print("="*40)
print()
# end of 1st commit

# 2nd commit start (missing value replaced with respected column mean)
data = data.fillna(data.mean())
# end of 2nd commit 


y=data.bphi
x=data.drop('bphi',axis=1)

m=x.shape[0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression as lm
model=lm().fit(x_train,y_train)

test = x_test.head(1)
predictions = model.predict(x_test.head(1))
import matplotlib.pyplot as plt
plt.scatter(y_test.head(1),predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")


predictions


predictions[0:1000]


print ("Score:", model.score(x_test, y_test))
print (predictions)
