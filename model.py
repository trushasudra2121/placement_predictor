import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.DataFrame({
    'cgpa':[6,7,8,9,7.5,8.7,9,7.6],
    'aptitude':[60,70,80,90,56,78,95,56],
    'communication':[6,5,7,8,5,6,7,9],
    'projects':[1,2,3,4,1,2,3,2],
    'placed':[0,1,1,1,0,1,1,0]
})

X = data[['cgpa',
    'aptitude',
    'communication',
    'projects']]
y = data['placed']

model = LogisticRegression()
model.fit(X,y)

pickle.dump(model, open('model.pkl', 'wb'))

print("model trained and saved")