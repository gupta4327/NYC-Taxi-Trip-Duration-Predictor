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
from dvclive import Live
import yaml
import pickle
from hyperopt import tpe, hp, Trials, STATUS_OK, fmin, space_eval
from hyperopt.pyll.base import scope
from src.utils import model_eval

#setting up infologger.info configuration 
from src.logger import infologger

infologger.info('Building a model')

#class for building features from dataset
class TrainModel:
    
    #initialising constructor
    def __init__(self, trainpath, testpath,model, feat, ohe_feat, seed, hyperparams,output_path, home_dir):

        self.model_cat = model
        self.onehot_encode_feat = ohe_feat
        self.trainpath = trainpath
        self.testpath = testpath 
        self.features = feat
        self.seed = seed
        self.hyperparams = hyperparams
        self.output_path = output_path
        self.home_dir = home_dir
        self.hyperopt_algo = tpe.suggest
        self.hyperopt_max_eval = 15
        self.trials = Trials()
        self.scoredic_list = []

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
        
        elif model == 'XtremeGradientBoost':

            self.model_instance = XGBRegressor

        else:
            infologger.info('Wrong model name has been passed')

    #function to read train and test data from designated path 
    def read_data(self):
        
        try:
            #try reading data from data path 
            self.df_train = pd.read_csv(self.trainpath)
            self.df_test = pd.read_csv(self.testpath)

        except Exception as e:

            #logging for error
            infologger.info(f'Reading of data from the paths has been failed with error :  {e}')

        else:

            #logging for succes
            infologger.info('Reading performed successfully')
    
    def feature(self):

        '''This function will keep only required features for modelling and dropoff other features and splits the data into input and output'''
        try:

            #keeping required features 
            self.df_train= self.df_train[self.features]
            self.df_test = self.df_test[self.features]

        except Exception as e:

            #logging for error
            infologger.info(f'Feature function failed with error :  {e}')

        else:

            #logging for succes
            infologger.info('Data with required features created successfully')
        
        #splitting in x and y 
        self.x_train = self.df_train.drop(columns=['trip_duration'])
        self.y_train = self.df_train['trip_duration']
        self.x_test = self.df_test.drop(columns= ['trip_duration'])
        self.y_test = self.df_test['trip_duration']

    def ohe(self):

        '''This function initializes a one hot encoder'''

        #initialing a one hot encoder to encode few features
        ohe = OneHotEncoder(handle_unknown='ignore')

        #a column transformer to perform ohe on passed onehot encoding fatures and passing the remaining features as it is 
        self.oh_trf = ColumnTransformer([('encode_ohe', ohe, self.onehot_encode_feat)],remainder='passthrough')

    
    def objective(self,params):

        '''This objective functon is used as a objective function for hyperopt objective function and also for fitting models 
        with different parameters and logging it to dvclive'''
        
        try:
        
               #initialising a model
            self.model = self.model_instance(random_state=self.seed,**params)

            #initialising a pipeline
            self.pipeline = Pipeline([('preprocess', self.oh_trf), ('regressor', self.model)])

            #fitting piepline on training data 
            self.pipeline.fit(self.x_train, self.y_train)

            #evaluating train and test score
            self.train_score = model_eval(self.pipeline, self.x_train, self.y_train)
            self.test_score = model_eval(self.pipeline, self.x_test, self.y_test)
            self.scoredic_list.append((self.train_score, self.test_score, params))

        except Exception as e:

            #logging for error
            infologger.info(f'Pipeline initialization and fitting has been failed with error :  {e}')

        else:

            #logging for succes
            infologger.info('Pipeline has been fitted successfully')

        return {'loss': self.test_score['Root Mean Square Error'], 'params':params, 'status':STATUS_OK}

    def hyperopt_finetune(self):
        
        try:
            #finding best hyperparameters for the model
            self.best = fmin(fn=self.objective, 
                        space=self.hyperparams, 
                        algo=self.hyperopt_algo, 
                        max_evals=self.hyperopt_max_eval, 
                        trials=self.trials)
            
            #getting a dictionary of best set of hyperparameter and fitting a model again with best hyperparameters and storing the model
            self.best_hyperparams = space_eval(self.hyperparams, self.best)
            self.model = self.model_instance(random_state=self.seed, **self.best_hyperparams)
            self.pipeline.fit(self.x_train, self.y_train)

        except Exception as e:

            #logging for error
            infologger.info(f'HyperParameter finetunning has been failed with error :  {e}')

        else:

            #logging for succes
            infologger.info('Hyperparameter finetunning has been successfully done')

    def livelog(self):

        try:


            with Live(self.output_path,save_dvc_exp=True) as live:
                

                for train_score, test_score , params in self.scoredic_list:

                    #live logging hyperparameters of model  
                    live.log_params(params)

                    #live logging all different scoring metrics of train data 
                    live.log_metric('train data/RMSE',train_score['Root Mean Square Error'])
                    live.log_metric('train data/RMSPE',train_score['Root Mean Square Percentage Error'])
                    live.log_metric('train data/MAE',train_score['Mean Absolute Error'])
                    live.log_metric('train data/R2_SCORE',train_score['R2 Score'])

                    #live logging all different scoring metrics of test data
                    live.log_metric('test data/RMSE',test_score['Root Mean Square Error'])
                    live.log_metric('test data/RMSPE',test_score['Root Mean Square Percentage Error'])
                    live.log_metric('test data/MAE',test_score['Mean Absolute Error'])
                    live.log_metric('test data/R2_SCORE',test_score['R2 Score']) 


    
        except Exception as e:

            #logging for error
            infologger.info(f'Live logging has been failed with error :  {e}')

        else:

            #logging for succes
            infologger.info('Logging to dvc live has been successfully done')

    
    def write_data(self):
        
        '''This function saves the data and model into destination folder'''

        try:
            
            # saving a binary model in models folder
            modelfilename = '/models/trained_models/' + self.model_cat +'_pipeline.pkl'
            pickle.dump(self.pipeline, open(Path(str(self.home_dir)+modelfilename),'wb'))

            #getting one hot encoded train and test data
            ohe_step = self.pipeline.named_steps['preprocess']
            df_train_transform  = ohe_step.transform(self.df_train)
            df_test_transform  = ohe_step.transform(self.df_test)
            
            #extracting feature nsames and converting numpy array into dataframe
            feature_names = ohe_step.get_feature_names_out()
            df_train_transform = pd.DataFrame(df_train_transform, columns=feature_names)
            df_test_transform = pd.DataFrame(df_test_transform, columns=feature_names)

            #naming and path to save the data
            traindatafilename = '/data/processed/train_data_model.csv'
            testdatafilename = '/data/processed/test_data_model.csv'
            traindatafilename =Path(str(self.home_dir)+traindatafilename)
            testdatafilename =Path(str(self.home_dir)+testdatafilename)
            
            #savng onehot encoded train test data as csv file
            df_train_transform.to_csv(traindatafilename)
            df_test_transform.to_csv(testdatafilename)       
            
        except Exception as e:
            
            # Log if error in saving a model and data
            infologger.info(f'Model and data saving has been failed with error: {e}')
        
        else:

            # Log if model saved successfully
            infologger.info('Model has been saved successfully')

    def train_model(self):

        '''Function to run for performing complete training process'''

        self.read_data()
        self.feature()
        self.ohe()
        self.hyperopt_finetune()
        self.livelog()
        infologger.info(f'Best hyperparameters of the models are : {self.best_hyperparams}')
        self.write_data()




#command line running arguments
@click.command()
@click.argument('traininput_filepath', type=click.Path())
@click.argument('testinput_filepath', type=click.Path())
@click.argument('hypparam_filepath', type=click.Path())
def main(traininput_filepath, testinput_filepath,hypparam_filepath ):

    """ This script trains a model and fine tunes the hyperparameters.
    """
    # Set up paths for input and output data and all other necessary files
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    
    #path for train and test data
    traininput_path = Path(data_dir.as_posix() + traininput_filepath)
    testinput_path = Path(data_dir.as_posix() + testinput_filepath)
    
    #dvclive for loggging experiments
    output_path = home_dir.as_posix() + '/dvclive'

    #loading parameters of train model from params.yaml file 
    params_path = Path(home_dir.as_posix()+'/params.yaml')
    model_params=yaml.safe_load(open(params_path))['train_model']
    model= model_params['model']
    features = model_params['features']
    ohe = model_params['ohe_features']
    seed = model_params['seed']
    
    #loading pickle files of hyperparameters
    hypparam_path = Path(home_dir.as_posix() + hypparam_filepath)
    hyperparams = pickle.load(open(hypparam_path, 'rb'))[model]
    
    #initating a object from TrainModel class
    trf = TrainModel(traininput_path, testinput_path,model, features, ohe, seed,hyperparams, output_path, home_dir)

    #training a model
    trf.train_model()

if __name__== "__main__":
    #call to a main function 
    main()