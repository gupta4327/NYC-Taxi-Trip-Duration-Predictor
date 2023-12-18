#importing necessary libraries and modules
import pandas as pd 
import numpy as np 
import click
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,mean_squared_log_error
from dvclive import Live
import yaml
import pickle
from hyperopt import tpe, hp, Trials, STATUS_OK, fmin, space_eval
from hyperopt.pyll.base import scope

#setting up logging.info configuration 
from src.logger import logging

logging.info('Building a model')

#class for building features from dataset
class TrainModel:
    def __init__(self, trainpath, testpath,model, feat, ohe_feat, hyperparams,output_path, home_dir):

        self.model_cat = model
        self.onehot_encode_feat = ohe_feat
        self.trainpath = trainpath
        self.testpath = testpath 
        self.features = feat
        self.hyperparams = hyperparams
        self.output_path = output_path
        self.home_dir = home_dir
        self.hyperopt_algo = tpe.suggest
        self.hyperopt_max_eval = 2
        self.trials = Trials()
        #logreg for logistic regression checking if model needs to be build is logistic regression
        if model == 'LinearRegression': 

            self.model_instance = LinearRegression
        
        #dt for decision trees checking if model needs to be build is decision trees
        elif model == 'DecisionTree':
            
            self.model_instance = DecisionTreeRegressor
        
        #rf for random forest checking if model needs to be build is Random Forest
        elif model == 'RandomForest':
            self.model_instance = RandomForestRegressor
        
        #gb for gradient boost cheking if model needs to be build is GradientBoost
        elif model == 'GradientBoost':
            self.model_instance = GradientBoostingRegressor
        
        elif model == 'XtremeGradietBoost':

            self.model_instance = XGBRegressor
   
    def read_data(self):
        self.df_train = pd.read_csv(self.trainpath)
        self.df_test = pd.read_csv(self.testpath)
    
    def feature(self):

        self.df_train= self.df_train[self.features]
        self.df_test = self.df_test[self.features]
        self.x_train = self.df_train.drop(columns=['trip_duration'])
        self.y_train = self.df_train['trip_duration']
        self.x_test = self.df_test.drop(columns= ['trip_duration'])
        self.y_test = self.df_test['trip_duration']

    def ohe(self):

        '''This function initializes a pipeline'''
        #initialing a one hot encoder to encode few features
        ohe = OneHotEncoder(handle_unknown='ignore')

        #a column transformer to perform ohe on passed onehot encoding fatures and passing the remaining features as it is 
        self.oh_trf = ColumnTransformer([('encode_ohe', ohe, self.onehot_encode_feat)],remainder='passthrough')

    def objective(self,params):

        self.model = self.model_instance(**params)

        self.pipeline = Pipeline([('preprocess', self.oh_trf), ('regressor', self.model)])

        self.pipeline.fit(self.x_train, self.y_train)

        self.train_score = model_eval(self.pipeline, self.x_train, self.y_train)
        self.test_score = model_eval(self.pipeline, self.x_test, self.y_test)

        with Live(self.output_path,save_dvc_exp=True) as live:
            live.log_params(params)

            live.log_metric('train data/RMSE',self.train_score['Root Mean Square Error'])
            live.log_metric('train data/RMSPE',self.train_score['Root Mean Square Percentage Error'])
            live.log_metric('train data/MAE',self.train_score['Mean Absolute Error'])
            live.log_metric('train data/R2_SCORE',self.train_score['R2 Score'])

            live.log_metric('test data/RMSE',self.test_score['Root Mean Square Error'])
            live.log_metric('test data/RMSPE',self.test_score['Root Mean Square Percentage Error'])
            live.log_metric('test data/MAE',self.test_score['Mean Absolute Error'])
            live.log_metric('test data/R2_SCORE',self.test_score['R2 Score'])
    
        return {'loss': self.test_score['Root Mean Square Error'], 'params':params, 'status':STATUS_OK}

    def hyperopt_finetune(self):
        
        
        self.best = fmin(fn=self.objective, 
                    space=self.hyperparams, 
                    algo=self.hyperopt_algo, 
                    max_evals=self.hyperopt_max_eval, 
                    trials=self.trials)
        
        self.best_hyperparams = space_eval(self.hyperparams, self.best)
        self.model = self.model_instance(**self.best_hyperparams)
        self.pipeline.fit(self.x_train, self.y_train)

    def write_data(self):
        '''This function writes the data into destination folder'''

        try:
            # saving a binary model in models folder
            modelfilename = '/models/trained_models/' + self.model_cat +'_pipeline.pkl'
            pickle.dump(self.pipeline, open(Path(str(self.home_dir)+modelfilename),'wb'))

            ohe_step = self.pipeline.named_steps['preprocess']
            
            df_train_transform  = ohe_step.transform(self.df_train)
            df_test_transform  = ohe_step.transform(self.df_test)
            
            feature_names = ohe_step.get_feature_names_out()

            df_train_transform = pd.DataFrame(df_train_transform, columns=feature_names)
            
            df_test_transform = pd.DataFrame(df_test_transform, columns=feature_names)

            traindatafilename = '/data/processed/train_data_model.csv'
            testdatafilename = '/data/processed/test_data_model.csv'

            traindatafilename =Path(str(self.home_dir)+traindatafilename)
            testdatafilename =Path(str(self.home_dir)+testdatafilename)
            
            df_train_transform.to_csv(traindatafilename)
            df_test_transform.to_csv(testdatafilename)

            
            
        except Exception as e:
            
            # Log if error in saving a model
            print(f'Model saving has been failed with error: {e}')
        
        else:
            # Log if model saved successfully
            print('Model has been saved successfully')

    def train_model(self):
        self.read_data()
        self.feature()
        self.ohe()
        self.hyperopt_finetune()
        print(self.best_hyperparams)
        self.write_data()


def model_eval(model, x, y):

    y_pred = model.predict(x)

    mse = round(mean_squared_error(y,y_pred),2)
        
        #root mean `square error
    rmse = round(np.sqrt(mean_squared_error(y,y_pred)),2)
    
    #mean absolute error
    mae = round(mean_absolute_error(y,y_pred),2)
    
    #root mean square percentage error
    rmspe = round(np.sqrt(np.sum(np.power(((y-y_pred)/y),2))/len(y)),3)
    
    #r2_score
    r2 = round(r2_score(y,y_pred),2)
    
    #adjusted_r2_score
    adjr2 = round(1-(1-r2_score(y,y_pred))*((x.shape[0]-1)/(x.shape[0]-x.shape[1]-1)),2)
    
    #dictionary storing all these testing score and this will be the returning value of function
    score_dict = {'Mean Square Error':mse, 'Root Mean Square Error':rmse,
                    'Mean Absolute Error':mae,'Root Mean Square Percentage Error':rmspe,'R2 Score':r2,
                    'Adjusted R2 Score': adjr2 }
    
    return score_dict

@click.command()
@click.argument('traininput_filepath', type=click.Path())
@click.argument('testinput_filepath', type=click.Path())
@click.argument('hypparam_filepath', type=click.Path())
def main(traininput_filepath, testinput_filepath,hypparam_filepath ):

    """ This script reads a train data from input folder and trains a model and store that model ito ../models folder.
    """
    # Set up paths for input and output data and all other necessary files
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    traininput_path = Path(data_dir.as_posix() + traininput_filepath)
    testinput_path = Path(data_dir.as_posix() + testinput_filepath)
    output_path = home_dir.as_posix() + '/dvclive'
    hypparam_path = Path(home_dir.as_posix() + hypparam_filepath)
    params_path = Path(home_dir.as_posix()+'/params.yaml')

    #loading parameters of train model from params.yaml file 
    model_params=yaml.safe_load(open(params_path))['train_model']
    model= model_params['model']
    features = model_params['features']
    ohe = model_params['ohe_features']

    #loading a hyperparameter file 
    hyperparams = pickle.load(open(hypparam_path, 'rb'))[model]
    # df_train = pd.read_csv(traininput_path)
    
    # df_test = pd.read_csv(testinput_path)

    # df_train = df_train[features]

    # df_test = df_test[features]

    # logging.info('call to class')
    
    #initating a object from TrainModel class
    trf = TrainModel(traininput_path, testinput_path,model, features, ohe, hyperparams, output_path, home_dir)

    trf.train_model()

    

if __name__== "__main__":
    #call to a main function 
    main()