# A Inference Model Repo Example

This is a formatted repo that is qualified to be deploy on AIbro inference API.

## Format Criteria

The repo should be structured in the following format:

repo <br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_[predict.py](#predict-py)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_[model](#model-and-data-folder)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_[data](#model-and-data-folder)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_[requirement.txt](#requirement-txt)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_[other artifacts](#other-artifacts)<br/>

### **predict.py**

This is the entrypoint that AIbro will call in the backend.

predict.py should contain two methods:

1. _load_model()_: this method should load and return your machine learning model from the "model" folder. An transformer-based Portuguese to English translator is used in this example repo.

```python
def load_model():
    # Portuguese to English translator
    translator = tf.saved_model.load('model')
    return translator
```

2. _run()_: this method used model as the input, load data from the "data" folder, predict, then return the inference result.

```python
def run(model):
    fp = open("./data/data.json", "r")
    data = json.load(fp)
    sentence = data["data"]

    result = {"data": model(sentence).numpy().decode("utf-8")}
    return result
```

**test tip**: predict.py() should be able to return an inference result by:

```python
run(load_model())
```

### **"model" and "data" folders**

There is no format restriction on the "model" and "data" folder as long as the input and output of load_model() and run() from predict.py are correct.

### **requirement.txt**

Before start deploying the model, packages from requirement.txt are installed to setup the environment.

### **Other artifacts**

all other files/folders.
