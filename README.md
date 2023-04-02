# MLLife

Package to use mlflow in Renault infrastructure. 

Metrics availables : 
- accuracy_score
- precision_score macro
- precision_score micro
- recall_score macro
- recall_score micro
- matthews_corrcoef
- f1_score macro
- f1_score micro
- roc_auc_score macro (if binary)
- roc_auc_score micro (if binary)


Visualisation availables : 
- confusion matrics
- classification reports
- roc curve (if binary)


# How to use the package ?
## 1. Setup 4 dictionnaries

#### Parameters Dictionnary  : 

Example of Parameters dictionnary : 
```python
dict_params = {
    "target_type" : "cs",
    "classification_method" : "multi",
    "model_version" : "v1.0"
}
```
You can specify any parameters you want (string type mandatory), but some are mandatory : 
- `classification_method` : specify if binary classification or not ("binary" vs "mutli"). Used to know how to apply metrics (roc curve, ...) 
- `model_version` : specify model version (ie. "v1.1", ...). Used to create a tag in mlflow. 
- `target_type` : specify name of target type (cs, gsfa, product, ...). Used for title name in visualisation. 


#### Artifacts Dictionnary : 

Example of artifacts path dictionnary : 
```python
dict_artifacts_paths = dict()
dict_artifacts_paths["ref"] = "mllife_test/data"
dict_artifacts_paths["models_classification"] = "mllife_test/model"
```
You can specify any path you want to store in artifacts. If you want to store your sklearn pipeline (set `bool_models` to True), you have to add in artifact path : 
- `models_classification` : folder path where to find *pipeline.joblib* file

#### Dataframe dictionnary :  
```python
dict_artifacts = {
    "df_train_with_predictions" : df_train,
    "df_val_with_predictions" : df_validation,
    "df_test_with_predictions" : df_test 
}
```
Dataframe to store into csv file 

#### Target vectors dictionnary : 
```python
dict_target_vectors = dict()
dict_target_vectors["y_train"] = df_train['target'].values
dict_target_vectors["y_train_pred"] = df_train['prediction'].values
dict_target_vectors["y_train_probas"] = df_train['probas'].values
dict_target_vectors["y_validation"] = df_validation['target'].values
dict_target_vectors["y_val_pred"] = df_validation['prediction'].values
dict_target_vectors["y_val_probas"] = df_validation['probas'].values
dict_target_vectors["y_test"] = df_test['target'].values
dict_target_vectors["y_test_pred"] = df_test['prediction'].values
dict_target_vectors["y_test_probas"] = df_test['probas'].values
```

You have to specify y_true, y_pred and y_proba for each dataframe (train, val and test) to get all metrics. 
If you don't have test set, you can create a dummy one : 

```python
import pandas as pd
import numpy
df_train = pd.DataFrame({"verbatim" : ["Hello world", "Bonjour monde", "Tutorial ici"], "target": ["A01", "B01", "C01"], "prediction": ["A01", "B01", "C03"], "probas": numpy.array([0.0, 1.0, 0.3])})
df_validation = pd.DataFrame({"verbatim" : ["Hello world", "Bonjour monde", "Tutorial ici"], "target": ["A01", "B01", "C01"], "prediction": ["A01", "B01", "C03"], "probas": numpy.array([0.0, 1.0, 0.3])})
df_test = pd.DataFrame({"verbatim" : ["Hello world", "Bonjour monde", "Tutorial ici"], "target": ["A01", "B01", "C01"], "prediction": ["A01", "B01", "C03"], "probas": numpy.array([0.0, 1.0, 0.3])})
```

## 2. Use Main classe

Run will be store in different experiment tabs. 

```python

from mllife.mlflow_process import MLflowProcess

mlflow_uri = "http://10.32.0.6:5000"
mlflow_experiment = "name_of_my_experiment_tab"
run_name = "name_of_my_run_name" 

mlf_process = MLflowProcess(mlflow_uri, mlflow_experiment)

mlf_process.mlflow_process(run_name,
                           dict_params = dict_params,
                           dict_target_vectors = dict_target_vectors,
                           dict_artifacts_datasets = dict_artifacts,
                           dict_artifacts_paths = dict_artifacts_paths,
                           bool_metrics = True,
                           bool_params = True,
                           bool_artifacts = True,
                           bool_models = False)
```