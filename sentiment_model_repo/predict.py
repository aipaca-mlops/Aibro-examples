# import important modules

import json  # load data
import joblib  # load model
from clean import text_cleaning  # function to clean the text


def load_model():
    # load model
    model = joblib.load("model/sentiment_model_pipeline.pkl")

    return model


def run(model):
    fp = open("data/data.json", "r")
    data = json.load(fp)
    review = text_cleaning(data["data"])

    result = {"data": model.predict([review])}
    return result


if __name__ == "__main__":
    run(load_model())
