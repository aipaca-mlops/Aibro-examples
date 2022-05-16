from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np


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
#     result = run(load_model())
#     print(result)
