from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np


#Load dataset
iris = datasets.load_iris()
X, y = iris['data'], iris['target']
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

np.savez("./data/iris.npz", X_test, )

model = RandomForestClassifier()
model.fit(X_train, y_train)

filename = './model/sklearn-rf.sav'
pickle.dump(model, open(filename, 'wb'))

def load_model():
    # Portuguese to English translator
    rf_model = pickle.load(open('model/sklearn-rf.sav', 'rb'))
    return rf_model

def run(model):
    data = np.load("./data/iris.npz")
    X_test = data['arr_0']

    result = {"data": model.predict(X_test)}
    return result

# if __name__ == "__main__":
#     run(load_model())
