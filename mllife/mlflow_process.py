# Imports
import os
import logging # To get logging messages
import mlflow
import mlflow.sklearn
import datetime
import requests
import numpy

import json # Temporary MLflow
from google.cloud import storage # Temporary MLflow
import ast # Temporary MLflow

from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score # To create metrics
from sklearn.metrics import classification_report # To create classification report
from sklearn.metrics import confusion_matrix # To compute confusion matrix
from sklearn.metrics import roc_curve # To compute ROC curve
from sklearn.metrics import RocCurveDisplay # To display ROC curve
import plotly # To create confusion matrix
import plotly.graph_objects as go # To create confusion matrix
import plotly.express as px # To display ROC curve plot
import pandas as pd

import joblib # To load classification pipeline


def upload_data(bucket_name, gcs_path, local_path):
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    
class MLflowProcess :
    """ 
    A custom class to help the use of mlflow.
    """ 

    def __init__(self, mlflow_uri, mlflow_experiment, with_server = True):
        """        
        Parameters
        ----------
        mlflow_uri : str
            Uri of mlflow server
        mlflow_experiment : str
            Name of experiment mlflow
        with_server : boolean
            Boolean to tell if you want to use the mlflow server
        """
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment = mlflow_experiment
        
        # Testing parameters
        if(not isinstance(mlflow_uri, str)):
            raise TypeError("mlflow_uri should be a string.")
        
        if(not isinstance(mlflow_experiment, str) or mlflow_experiment == ""):
            raise TypeError("mlflow_experiment should be a non-empty string.")
        
        if(not isinstance(with_server, bool)):
            raise TypeError("with_server should be a boolean.")
        
        if(with_server == True):
            try :
                # Test if the server can be reached or not before everything else
                requests.get(mlflow_uri, timeout = 5)
            except : 
                raise RuntimeError("Can't access the mlflow server. Either the server isn't up or the URI given is wrong : " + mlflow_uri)
            
            # Set the config values
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.mlflow_experiment)
        else :
            logging.info("Process not connected to the mlflow server.")
    
    
    def custom_start_run(self, run_name):
        """ 
        Method to start the mlflow run
        
        Parameters
        ----------
        run_name : str
            Name of the run
            
        """
        # Testing parameters
        if(not isinstance(run_name, str) or run_name == ""):
            raise TypeError("run_name should be a non-empty string.")
        
        # Run start
        logging.info("Start MLflow run : " + run_name)
        mlflow.start_run(run_name = run_name)
        
    def custom_end_run(self):
        """ 
        Method to end the mlflow run
        """
        # Run end
        mlflow.end_run()
        logging.info("Current MLflow run ended.")
        
    def add_tags(self,tags):
        """Method to add tags inside experiment
            
        Parameters
        ----------
        tags : dict(str)
            dictionnary of value to add in tags
       
        """
        mlflow.set_tags(tags)
    
    
    def add_time_name(self, run_name):
        """ Method to add datetime inside run_name
            
        Parameters
        ----------
        run_name : str
            Name of the run (usually using parameters used)
       
        Returns
        ------
        str
            A string containing a part of the datetime (year_month)
        str
            A string containing a part of the datetime (day)
        str
            A string containing the hour time and the run_name

        """
        if(run_name == "" or not isinstance(run_name, str)):
            raise ValueError("run_name should be a non-empty string.")
        
        dateTimeObj = datetime.datetime.now()
        year_month_string = str(dateTimeObj.year) + "_" + str(dateTimeObj.month)
        day_string = str(dateTimeObj.day)
        hour_string = str(dateTimeObj.hour+2) + '_' + str(dateTimeObj.minute)
        timed_run_name = hour_string + "x" + run_name
        return year_month_string, day_string, timed_run_name
        
        
    def eval_metrics(self, actual, pred, probas, target_vector_type, target_type):
        """ 
        Method to create the metrics to send to the mlflow server
        
        Parameters
        ----------
        actual : array
            Actual target vector
        pred : array
            Predicted target vector
        probas : array
            Probas of highest predicted target
        target_vector_type : str
            Type of target vector (either "train", "validation" or "test")
        target_type : str
            Target type to predict (should be the class to predict in a binary classification problem)
        
        Returns
        ------
        dict(str, float)
            A dictionary containing the values for each metric
        """
        # Testing parameters
        if(not isinstance(actual, numpy.ndarray) or actual.size == 0):
            raise TypeError("actual should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in actual)): # or len(elem2) == 0 for elem2 in actual): / Value empty in the middle to test
            raise ValueError("Values in actual should not be empty and should be of type 'str'")
        
        if(not isinstance(pred, numpy.ndarray) or pred.size == 0):
            raise TypeError("pred should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in pred)): # or len(elem) == 0 for elem in pred):
            raise ValueError("Values in pred should not be empty and should be of type 'str'")
        
        if(not isinstance(probas, numpy.ndarray) or probas.size == 0):
            raise TypeError("probas should be a non-empty array.")
        
        if(not all(isinstance(elem, numpy.floating) for elem in probas)): # or len(elem) == 0 for elem in pred):
            raise ValueError("Values in probas should not be empty and should be of type 'float'")
        
        if(len(actual) != len(pred)):
            raise ValueError ("Found input variables (actual and pred) with inconsistent numbers of samples")
        
        if(len(actual) != len(probas)):
            raise ValueError ("Found input variables (actual, pred and probas) with inconsistent numbers of samples")
        
        if(not isinstance(target_vector_type, str)):
            raise TypeError ("target_vector_type should be a string")
            
        if(target_vector_type != "train" and target_vector_type != "validation" and target_vector_type != "test"):
            raise ValueError ("target_vector_type should either be 'train', 'validation' or 'test'")

        if(not isinstance(target_type, str)):
            raise TypeError ("target_type should be a string")
        
        # Dictionary creation
        metrics = dict()
        metrics["acc__"+target_vector_type] = accuracy_score(actual, pred)
        metrics["pre__"+target_vector_type] = precision_score(actual, pred, average = 'macro')
        metrics["pre_micro__"+target_vector_type] = precision_score(actual, pred, average = 'micro')
        metrics["rec__"+target_vector_type] = recall_score(actual, pred, average = 'macro')
        metrics["rec_micro__"+target_vector_type] = recall_score(actual, pred, average = 'micro')
        metrics["mcc__"+target_vector_type] = matthews_corrcoef(actual, pred)
        metrics["f1__"+target_vector_type] = f1_score(actual, pred, average = 'macro')
        metrics["f1_micro__"+target_vector_type] = f1_score(actual, pred, average = 'micro')
        
        if(len(numpy.unique(actual)) == 2):
            probas = numpy.where(pred == target_type, probas, 1-probas)
            actual_01 = numpy.where(actual == target_type, 1, 0)
            metrics["auc__"+target_vector_type] = roc_auc_score(actual_01.astype(int), probas, average = 'macro')
            metrics["auc_micro__"+target_vector_type] = roc_auc_score(actual_01.astype(int), probas, average = 'micro')

        logging.info("Metrics successfully created")
        
        return metrics
    
    
    def get_all_metrics(self, dict_target_vectors, dict_params):
        """ 
        Method to merge all the metrics into a single dictionary
        
        Parameters
        ----------
        dict_target_vectors : dict(str : array)
            Dictionary of target vectors
        dict_params : dict(str : str)
            Dictionary of parameters
        
        Returns
        -------
        dict(str, float)
            A dictionary containing the values for each metric
        """
        # Testing parameters
        if(dict_target_vectors == {}):
            raise ValueError("dict_target_vectors should not be empty.")
            
        for param_key, param_value in dict_target_vectors.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_target_vectors.")
            elif(not isinstance(param_value, numpy.ndarray)):
                raise ValueError("Wrong type of values in dict_target_vectors.")
        
        if(dict_params == {}):
            raise ValueError("dict_params should not be empty.")
            
        for param_key, param_value in dict_params.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_params.")
            elif(not isinstance(param_value, str) and not all(isinstance(elem, str) for elem in param_value)):
                raise ValueError("Wrong type of values in dict_params.")
            elif(param_value == "" or param_value[0] == ""):
                raise ValueError("Can't have an empty string as value in dict_params.")
        
        
        # Get all metrics
        metrics_train = self.eval_metrics(dict_target_vectors["y_train"], dict_target_vectors["y_train_pred"], dict_target_vectors["y_train_probas"], "train", dict_params["target_type"])
        metrics_val = self.eval_metrics(dict_target_vectors["y_validation"], dict_target_vectors["y_val_pred"],dict_target_vectors["y_val_probas"], "validation", dict_params["target_type"])
        metrics_test = self.eval_metrics(dict_target_vectors["y_test"], dict_target_vectors["y_test_pred"],dict_target_vectors["y_test_probas"], "test", dict_params["target_type"])
        
        # Merge metrics
        all_metrics = {**metrics_train, **metrics_val, **metrics_test}
        return all_metrics
        
    
    def create_classification_report(self, inv_actual, inv_pred, target_vector_type):
        """ 
        Method to create the classification report
        
        Parameters
        ----------
        inv_actual : array
            Inverse transformed actual target vector
        inv_pred : array
            Inverse transformed predicted target vector
        target_vector_type : string
            Type of the target vectors ("train", "val" or "test")

        """
        
        # Testing parameters
        if(not isinstance(inv_actual, numpy.ndarray) or inv_actual.size == 0):
            raise TypeError("inv_actual should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in inv_actual)): # or len(elem2) == 0 for elem2 in inv_actual): / Value empty in the middle to test
            raise ValueError("Values in inv_actual should not be empty and should be of type 'str'")
        
        if(not isinstance(inv_pred, numpy.ndarray) or inv_pred.size == 0):
            raise TypeError("inv_pred should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in inv_pred)): # or len(elem) == 0 for elem in inv_pred):
            raise ValueError("Values in pred should not be empty and should be of type 'str'")
        
        if(len(inv_actual) != len(inv_pred)):
            raise ValueError ("Found input variables (inv_actual and inv_pred) with inconsistent numbers of samples")
        
        if(not isinstance(target_vector_type, str)):
            raise TypeError ("target_vector_type should be a string")
            
        if(target_vector_type != "train" and target_vector_type != "validation" and target_vector_type != "test"):
            raise ValueError ("target_vector_type should either be 'train', 'validation' or 'test'")
        
        classif_report = classification_report(inv_actual, inv_pred, output_dict=True)
        df_classif_report = pd.DataFrame(classif_report).transpose()
        os.makedirs(os.path.dirname("mlflow_artifacts/classification_reports/"), exist_ok=True)
        pd.DataFrame(df_classif_report).to_csv('mlflow_artifacts/classification_reports/classification_report_'+ target_vector_type +'.csv', index=True, sep = ";", encoding = "utf-8")
        
        
    def create_confusion_matrix(self, inv_actual, inv_pred, target_vector_type, normalize):
        """ 
        Method to create the confusion matrix
        
        Parameters
        ----------
        inv_actual : array
            Inverse transformed actual target vector
        inv_pred : array
            Inverse transformed predicted target vector
        target_vector_type : string
            Type of the target vectors ("train", "val" or "test")
        normalize : str
            Option to normalize or not for the confusion matrix
        """
        
        # Testing parameters
        if(not isinstance(inv_actual, numpy.ndarray) or inv_actual.size == 0):
            raise TypeError("inv_actual should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in inv_actual)): # or len(elem2) == 0 for elem2 in inv_actual): / Value empty in the middle to test
            raise ValueError("Values in inv_actual should not be empty and should be of type 'str'")
        
        if(not isinstance(inv_pred, numpy.ndarray) or inv_pred.size == 0):
            raise TypeError("inv_pred should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in inv_pred)): # or len(elem) == 0 for elem in inv_pred):
            raise ValueError("Values in pred should not be empty and should be of type 'str'")
        
        if(len(inv_actual) != len(inv_pred)):
            raise ValueError ("Found input variables (inv_actual and inv_pred) with inconsistent numbers of samples")
            
        if(not isinstance(target_vector_type, str)):
            raise TypeError ("target_vector_type should be a string")
            
        if(target_vector_type != "train" and target_vector_type != "validation" and target_vector_type != "test"):
            raise ValueError ("target_vector_type should either be 'train', 'validation' or 'test'")
            
        if(normalize not in [None, "true"]):
            raise TypeError ("target_vector_type should be a string")
        
        conf_matrix = confusion_matrix(inv_actual, inv_pred, normalize=normalize)
        # conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        classes = [c for c in set(list(inv_actual) + list(inv_pred))]
        classes.sort()
        
        fig = go.Figure(data=go.Heatmap(
                           z=conf_matrix,
                           x=classes,
                           y=classes,
                           hoverongaps = False))
        
        os.makedirs(os.path.dirname("mlflow_artifacts/confusion_matrices/"), exist_ok=True)
        plotly.offline.plot(fig, filename = "mlflow_artifacts/confusion_matrices/confusion_matrix_" + target_vector_type + ".html")

    def create_roc_curve(self, actual, pred, probas, target_vector_type, target_type):
        """ 
        Method to create the ROC curve
        
        Parameters
        ----------
        actual : array
            Actual target vector
        pred : array
            Predicted target vector
        probas : array
            Probas of highest predicted target
        target_vector_type : str
            Type of target vector (either "train", "validation" or "test")
        target_type : str
            Target type to predict (should be the class to predict in a binary classification problem)
        """
        
        # Testing parameters
        if(not isinstance(actual, numpy.ndarray) or actual.size == 0):
            raise TypeError("actual should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in actual)): # or len(elem2) == 0 for elem2 in actual): / Value empty in the middle to test
            raise ValueError("Values in actual should not be empty and should be of type 'str'")
        
        if(not isinstance(pred, numpy.ndarray) or pred.size == 0):
            raise TypeError("pred should be a non-empty array.")
        
        if(not all(isinstance(elem, str) for elem in pred)): # or len(elem) == 0 for elem in pred):
            raise ValueError("Values in pred should not be empty and should be of type 'str'")
        
        if(not isinstance(probas, numpy.ndarray) or probas.size == 0):
            raise TypeError("probas should be a non-empty array.")
        
        if(not all(isinstance(elem, numpy.floating) for elem in probas)): # or len(elem) == 0 for elem in pred):
            raise ValueError("Values in probas should not be empty and should be of type 'float'")
        
        if(len(actual) != len(pred)):
            raise ValueError ("Found input variables (actual and pred) with inconsistent numbers of samples")
        
        if(len(actual) != len(probas)):
            raise ValueError ("Found input variables (actual, pred and probas) with inconsistent numbers of samples")
        
        if(not isinstance(target_vector_type, str)):
            raise TypeError ("target_vector_type should be a string")
            
        if(target_vector_type != "train" and target_vector_type != "validation" and target_vector_type != "test"):
            raise ValueError ("target_vector_type should either be 'train', 'validation' or 'test'")

        if(not isinstance(target_type, str)):
            raise TypeError ("target_type should be a string")
        
        probas = numpy.where(pred == target_type, probas, 1-probas)
        actual_01 = numpy.where(actual == target_type, 1, 0)

        fpr, tpr, _ = roc_curve(actual, probas, pos_label = target_type)

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={roc_auc_score(actual_01.astype(int), probas):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500)
        
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)
        
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        
        os.makedirs(os.path.dirname("mlflow_artifacts/roc_curves/"), exist_ok=True)
        plotly.offline.plot(fig, filename = "mlflow_artifacts/roc_curves/roc_curve_" + target_vector_type + ".html")
        
        
    def save_params(self, dict_params):
        """ 
        Method to save the parameters to the mlflow server
        
        Parameters
        ----------
        dict_params : dict(str : str)
            Dictionary of parameters
            
        """
        # Testing parameters
        if(dict_params == {}):
            raise ValueError("dict_params should not be empty.")
        
        for param_key, param_value in dict_params.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_params.")
            elif(not isinstance(param_value, str) and not all(isinstance(elem, str) for elem in param_value)):
                raise ValueError("Wrong type of values in dict_params.")
            elif(param_value == "" or param_value[0] == ""):
                raise ValueError("Can't have an empty string as value in dict_params.")
        
        logging.info("Saving parameters to mlflow...")
        
        # Send parameters
        mlflow.log_params(dict_params)
        logging.info("Finished saving parameters to mlflow.")
    

    def save_metrics(self, dict_metrics):
        """ 
        Method to save the metrics to the mlflow server
        
        Parameters
        ----------
        dict_metrics : dict(str : float)
            Dictionary of metrics

        """
        # Testing parameters
        if(dict_metrics == {}):
            raise ValueError("dict_target_vectors should not be empty.")
            
        for param_key, param_value in dict_metrics.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_metrics.")
            elif(not isinstance(param_value, float)):
                raise ValueError("Wrong type of values in dict_metrics.")
            
        logging.info("Saving metrics to mlflow...")
        
        # Send metrics
        mlflow.log_metrics(dict_metrics)
        logging.info("Finished saving metrics to mlflow.")

    
    def save_artifacts(self, dict_artifacts_paths):
        """ 
        Method to save the artifacts to the mlflow server
        
        Parameters
        ----------
        dict_artifacts_paths : dict(str : str)
            Paths of artifacts that you want to save
        
        """
        # Testing parameters
        if(dict_artifacts_paths == {}):
            raise ValueError("dict_artifact_paths should not be empty.")
        
        for param_key, param_value in dict_artifacts_paths.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_artifacts_paths.")
            elif(not isinstance(param_value, str)):
                raise ValueError("Wrong type of values in dict_artifacts_paths.")
        
        logging.info("Saving artifacts to mlflow...")
        
        # Send artifacts
        counter_correct_paths = 0
        for name_artifact_dir, artifact_dir in dict_artifacts_paths.items() :
            logging.info("Saving artifact directory : " + artifact_dir)
            
            try : 
                list_artifacts = os.listdir(artifact_dir)
                counter_correct_paths += 1
            except FileNotFoundError : 
                logging.info("Artifact directory not found : " + artifact_dir)
                continue
            
            logging.info("List of artifacts : " + str(list_artifacts))
            for artifact in list_artifacts :
                local_path = artifact_dir + '/' + artifact
                
                if(name_artifact_dir == "embeddings_model"):
                    logging.info("Saving embeddings model...")
                    mlflow.log_artifacts(local_path, artifact_path = local_path)
                
                elif(os.path.isdir(local_path)):
                    logging.info("Saving local directory : " + local_path)
                    mlflow.log_artifacts(local_path, artifact_path = local_path)
                else :
                    logging.info("Saving artifact : " + local_path)
                    mlflow.log_artifact(local_path, artifact_path = artifact_dir)
        
        if(counter_correct_paths == 0):
            raise FileNotFoundError("No correct path in dict_artifacts_paths.")
        
        logging.info("Finished saving artifacts to mlflow.")
        
        
    def save_classification_model(self, classif_model_path):
        """ Method to save the classification_model to the mlflow server
        
        Parameters
        ----------
        classif_model_path : str
            Local path to the classification model (.joblib)

        """
        # Testing parameters
        if(not isinstance(classif_model_path, str) or classif_model_path == ""):
            raise TypeError("classif_model_path should be a non-empty string.")
            
        logging.info("Saving classification model to mlflow...")
        
        # Log model
        pipeline_joblib = joblib.load(classif_model_path + '/pipeline.joblib')
        mlflow.sklearn.log_model(sk_model = pipeline_joblib, artifact_path = "models")
        
        logging.info("Finished classification model to mlflow.") 
    
    def mlflow_process(self, run_name, dict_params = None, dict_target_vectors = None, dict_artifacts_datasets = None, dict_artifacts_paths = None, 
                       bool_params = False, bool_metrics = False, bool_artifacts = False, bool_models = False):
        """ Main method that does all the saving steps
        
        Parameters
        ----------
        run_name : str
            Name of the run (usually using parameters used)
        dict_params : dict(str : str)
            Dictionary of parameters
        dict_target_vectors : dict(str : array)
            Dictionary of target vectors
        dict_artifacts_datasets : dict(str : array)
            Dictionary of the datasets used as artifacts
        dict_artifacts_paths : dict(str : str)
            Paths of artifacts that you want to save
        bool_params : boolean
            Boolean to indicate if we want to save parameters
        bool_metrics : boolean
            Boolean to indicate if we want to save metrics
        bool_artifacts : boolean
            Boolean to indicate if we want to save artifacts
        bool_models : boolean
            Boolean to indicate if we want to save models
        
        """
        
        # Testing parameters
    
        logging.info("Saving parameters, metrics and artifacts to MLflow...")
        
        # Testing parameters
        if(run_name == "" or not isinstance(run_name, str)):
            raise ValueError("run_name should be a non-empty string.")
        
        if(dict_params == {}):
            raise ValueError("dict_params should not be empty.")
        for param_key, param_value in dict_params.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_params.")
            elif(not isinstance(param_value, str) and not all(isinstance(elem, str) for elem in param_value)):
                raise ValueError("Wrong type of values in dict_params.")
            elif(param_value == "" or param_value[0] == ""):
                raise ValueError("Can't have an empty string as value in dict_params.")
        
        if(dict_target_vectors == {}):
            raise ValueError("dict_target_vectors should not be empty.")
        for param_key, param_value in dict_target_vectors.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_target_vectors.")
            elif(not isinstance(param_value, numpy.ndarray) and not all(isinstance(elem, str) for elem in param_value)):
                raise ValueError("Wrong type of values in dict_target_vectors.")
        
        if(dict_artifacts_datasets == {}):
            raise ValueError("dict_artifacts_datasets should not be empty.")
        for param_key, param_value in dict_artifacts_datasets.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_artifacts_datasets.")
            elif(not isinstance(param_value, numpy.ndarray) and not all(isinstance(elem, str) for elem in param_value)):
                raise ValueError("Wrong type of values in dict_artifacts_datasets.")
        
        if(dict_artifacts_paths == {}):
            raise ValueError("dict_artifacts_paths should not be empty.")
        for param_key, param_value in dict_artifacts_paths.items() :
            if(not isinstance(param_key, str)):
                raise KeyError("Wrong type of keys in dict_artifacts_paths.")
            elif(not isinstance(param_value, str) and not all(isinstance(elem, str) for elem in param_value)):
                raise ValueError("Wrong type of values in dict_artifacts_paths.")
            elif(param_value == "" or param_value[0] == ""):
                raise ValueError("Can't have an empty string as value in dict_artifacts_paths.")
        
        if(not isinstance(bool_params, bool) or not isinstance(bool_metrics, bool) or not isinstance(bool_artifacts, bool) or not isinstance(bool_models, bool)):
            raise TypeError("bool_params, bool_metrics, bool_artifacts and bool_models should be booleans.")
        
        
        # Start run
        year_month_string, day_string, timed_run_name = self.add_time_name(run_name)
        self.custom_start_run(year_month_string + "/" + day_string + "/" + timed_run_name)
    
        # Params
        if (bool_params == True):
            self.save_params(dict_params)
            
        # Metrics
        if (bool_metrics == True):
            dict_metrics = self.get_all_metrics(dict_target_vectors,dict_params)
            self.save_metrics(dict_metrics)
            
        # Artifacts
        if (bool_artifacts == True):
            # Create classification reports
            self.create_classification_report(dict_target_vectors["y_train"], dict_target_vectors["y_train_pred"], "train")
            self.create_classification_report(dict_target_vectors["y_validation"], dict_target_vectors["y_val_pred"], "validation")
            self.create_classification_report(dict_target_vectors["y_test"], dict_target_vectors["y_test_pred"], "test")

            # Create confusion matrix
            self.create_confusion_matrix(dict_target_vectors["y_train"], dict_target_vectors["y_train_pred"], "train", "true")
            self.create_confusion_matrix(dict_target_vectors["y_validation"], dict_target_vectors["y_val_pred"], "validation", "true")
            self.create_confusion_matrix(dict_target_vectors["y_test"], dict_target_vectors["y_test_pred"], "test", None)
            
            if dict_params["classification_method"] ==  "binary" : 
            # Create roc curves
                self.create_roc_curve(dict_target_vectors["y_train"], dict_target_vectors["y_train_pred"], dict_target_vectors["y_train_probas"],"train",dict_params["target_type"])
                self.create_roc_curve(dict_target_vectors["y_validation"], dict_target_vectors["y_val_pred"], dict_target_vectors["y_val_probas"],"validation",dict_params["target_type"])
                self.create_roc_curve(dict_target_vectors["y_test"], dict_target_vectors["y_test_pred"], dict_target_vectors["y_test_probas"],"test",dict_params["target_type"])
        
            # Create data to .csv
            os.makedirs(os.path.dirname("mlflow_artifacts/df_predictions/"), exist_ok=True)
            pd.DataFrame(dict_artifacts_datasets["df_train_with_predictions"]).to_csv('mlflow_artifacts/df_predictions/df_train_predictions.csv', index=False, sep = ";", encoding = "utf-8")
            pd.DataFrame(dict_artifacts_datasets["df_val_with_predictions"]).to_csv('mlflow_artifacts/df_predictions/df_val_predictions.csv', index=False, sep = ";", encoding = "utf-8")
            pd.DataFrame(dict_artifacts_datasets["df_test_with_predictions"]).to_csv('mlflow_artifacts/df_predictions/df_test_predictions.csv', index=False, sep = ";", encoding = "utf-8")
            
            dict_other_paths = {
                "path_classification_reports" : "mlflow_artifacts/classification_reports",
                "path_confusion_matrices" : "mlflow_artifacts/confusion_matrices",
                "path_data_predictions" : "mlflow_artifacts/df_predictions"
            }
            
            if dict_params["classification_method"] ==  "binary" : # add roc curve
                dict_other_paths.update({"path_roc_curves" : "mlflow_artifacts/roc_curves"})
            
            dict_artifacts_paths.update(dict_other_paths)
            
            dict_not_needed_keys = dict_artifacts_paths.copy()
            not_needed_keys = []#["models_classification"]
            for key in not_needed_keys :
                dict_not_needed_keys.pop(key)
            
            self.save_artifacts(dict_not_needed_keys)
            
        # Classification model     
        if (bool_models == True):
            self.save_classification_model(dict_artifacts_paths["models_classification"])
            
                
        self.add_tags({"version": dict_params["model_version"]})
        self.custom_end_run()
        logging.info("MLflow process completed.")

