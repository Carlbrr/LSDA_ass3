import pandas as pd
import mlflow

## NOTE: You can use Microsoft Azure Machine Learning Studio for experiment tracking. Follow assignment description and uncomment below for that (you might also need to pip azureml (pip install azureml-core):
from azureml.core import Workspace
##ws = Workspace.from_config()
ws = Workspace(subscription_id = "aabeddb0-41f5-4bcc-85e9-94af5d2928f5", resource_group = "myVM_group", workspace_name = "ML-ws", auth=None, _location=None, _disable_service_check=False, _workspace_id=None, sku='basic', tags=None, _cloud='AzureCloud')

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri()) #uncomment

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

# mlflow.set_tracking_uri("http://training.itu.dk:5000/")

# TODO: Set the experiment name
mlflow.set_experiment("first")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="<testing>"):
    # TODO: Insert path to dataset
    df = pd.read_json("./dataset.json", orient="split")
##label encode the directions first, such that we can get a mean of the directions, as we will loose data if we try to mea
#n string values.
labelEnconder = LabelEncoder()
dirColumn = df["Direction"]
df["Direction"] = labelEnconder.fit_transform(dirColumn) 


##group by 3H time interval
df = pd.DataFrame(df.groupby("Source_time").mean(), columns=df.columns)
df = df.drop(['Source_time'], axis= 1)

##fit transform on dataframe returns a numpy array, so we cant call dropna on the returned dataframe, as it doesnt know the method.

class MissingValues(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        df = pd.DataFrame(X)
        df.dropna(inplace=True)
        return df

#wind directions

class AddWindDirections(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        df = pd.DataFrame(X)
        le = LabelEncoder()
        directionsColumn = df["Direction"]
        df["Direction"] = le.fit_transform(directionsColumn)        
        return df
        
#now the data shouldve been correctly preproccesed. 
##NOTE TO SELF: we could try to handle missing values differently? 
##Another could be to try hot label encoder instead of label encoder
##Different regression models of.
##What about a different scaler? is that a thing?
##Mean Squared Error (MSE). Root Mean Squared Error (RMSE). Mean Absolute Error (MAE). These are the three error metrics
#most commonly used to evaluate performance of a regression model. - what about r2 score? 
##Estimator score method: Estimators have a score method providing a default evaluation criterion for the problem they are designed to solve
#https://scikit-learn.org/stable/modules/model_evaluation.html
##

pipeline = Pipeline(steps=[('MissingValues',MissingValues()),('AddWindDirections',AddWindDirections()),('Scaler', MinMaxScaler()),('RegModel', LinearRegression())])

# TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
metrics = [
    ("MAE", mean_absolute_error, []),
]

X = df[["Speed","Direction"]]
y = df["Total"]

number_of_splits = 2

#TODO: Log your parameters. What parameters are important to log?
#HINT: You can get access to the transformers in your pipeline using `pipeline.steps`

for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
    pipeline.fit(X.iloc[train],y.iloc[train])
    predictions = pipeline.predict(X.iloc[test])
    truth = y.iloc[test]

    from matplotlib import pyplot as plt 
    plt.plot(truth.index, truth.values, label="Truth")
    plt.plot(truth.index, predictions, label="Predictions")
    plt.show()

    # Calculate and save the metrics for this fold
    for name, func, scores in metrics:
        score = func(truth, predictions)
        scores.append(score)

# Log a summary of the metrics
for name, _, scores in metrics:
        # NOTE: Here we just log the mean of the scores. 
        # Are there other summarizations that could be interesting?
        mean_score = sum(scores)/number_of_splits
        mlflow.log_metric(f"mean_{name}", mean_score)
