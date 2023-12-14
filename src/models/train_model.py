#importing necessary libraries and modules
import pandas as pd 
import numpy as np 
import click
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import yaml
import pickle


#setting up logging configuration 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('train_model.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('Creating dataset for modelling')

#class for building features from dataset
class TrainModel:

    def __init__(self, read_path,model, features, ohe_feat, home_dir, **kwargs):

        self.read_path = read_path
        self.features = features
        self.model_cat = model
        self.onehot_encode_feat = ohe_feat
        self.home_dir = home_dir
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

        try:        
            self.model = self.model_instance(**kwargs)   #model initialisation
        
        except Exception as e:     

            logger.info(f'Model initialisation failed with error : {e}')   #custom expectation raising 

    
    def read_data(self):

        '''This function reads csv data from input path and stores it into a dataframe'''

        try:
            # Read data from the provided CSV file
            self.df = pd.read_csv(self.read_path)
        except Exception as e:
            # Log if reading fails
            logger.info(f'Reading failed with error: {e}')
        else:
            # Log if reading is successful
            logger.info('Read performed successfully')

    def keep_features(self):

        try:
            self.df = self.df[self.features]

        except Exception as e:
            # Log if reading fails
            logger.info(f'Feature selecting failed with error: {e}')
        
        else:
            logger.info('feature has been selected successfully')

    def pipeline(self):

        '''This function initializes a pipeline'''
        
        try:
            #initialing a one hot encoder to encode few features
            ohe = OneHotEncoder(handle_unknown='ignore')

            #a column transformer to perform ohe on passed onehot encoding fatures and passing the remaining features as it is 
            self.oh_trf = ColumnTransformer([('encode_ohe', ohe, self.onehot_encode_feat)],remainder='passthrough')

            #initializing a pipeline
            self.model_pipeline = Pipeline([('preprocess', self.oh_trf), ('regressor', self.model)])

        except Exception as e:

            #logger if pipeline initialization fails 
            logger.info(f'Pipeline initialisation failed with error : {e}')

        else:

            #log if everything goes fine
            logger.info('Pipeline has been initialised successfully')
    
    def write_data(self):

        '''This function writes the data into destination folder'''

        try:
            # saving a binary model in models folder
            filename = '/models/trained_pipeline.pkl'
            pickle.dump(self.model_pipeline, open(Path(str(self.home_dir)+filename),'wb'))
            
        except Exception as e:
            
            # Log if error in saving a model
            logger.info(f'Model saving has been failed with error: {e}')
        
        else:
            # Log if model saved successfully
            logger.info('Model has been saved successfully')

    
    def input_output_split(self):

        #seperating input output columns from the data
        self.x_train = self.df.drop(columns=['trip_duration'])
        self.y_train = self.df['trip_duration']

    def pipeline_fit(self):
        '''This function performs all the operation in sync from reading a data from path to training a model. Post initialing a object 
        making a call to pipeline_fit function will perform all the operations till saving a model automatically'''

        try:
            #calling all the functions in sequence to train a model
            self.read_data()
            self.keep_features()
            self.input_output_split()
            self.pipeline()
            self.model_pipeline.fit(self.x_train, self.y_train)
        
        except Exception as e:

            #log if model training is failed 
            logger.info(f'Pipeline fit on train data has been failed with error : {e}')

        else:
            
            #logs if mopdel training has been performed successfully 
            logger.info('Pipeline has been fitted successfully')
            self.write_data()
            return self.model_pipeline

@click.command()
@click.argument('input_filepath', type=click.Path())
def main(input_filepath):

    """ This script reads a train data from input folder and trains a model and store that model ito ../models folder.
    """
    # Set up paths for input and output data and all other necessary files
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    input_path = Path(data_dir.as_posix() + input_filepath)
    params_path = Path(home_dir.as_posix()+'/params.yaml')

    #loading parameters of train model from params.yaml file 
    model_params=yaml.safe_load(open(params_path))['train_model']
    model= model_params['model']
    features = model_params['features']
    ohe = model_params['ohe_features']
    hyperparams = yaml.safe_load(open(params_path))['train_model']['hyperparameters']

    #initating a object from TrainModel class
    model = TrainModel(input_path, model,features, ohe, home_dir,**hyperparams)

    #function call to fit and save a model 
    model.pipeline_fit()

if __name__== "__main__":
    #call to a main function 
    main()