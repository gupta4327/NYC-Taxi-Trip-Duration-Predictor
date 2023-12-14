import pandas as pd 
import numpy as np 
import click
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,mean_squared_log_error
import yaml
from dvclive import Live
import matplotlib.pyplot as plt

class ModelEvaluation:

    def __init__(self, x,y,model,params,split, output_path):
        self.x = x
        self.y = y 
        self.model = model
        self.params = params
        self.split = split
        self.output_path = output_path
    

    def model_scores(self):

        self.y_pred = self.model.predict(self.x)    

          #mean_square_error
        self.mse = round(mean_squared_error(self.y,self.y_pred),2)
        
        #root mean square error
        self.rmse = round(np.sqrt(mean_squared_error(self.y,self.y_pred)),2)
        
        #mean absolute error
        self.mae = round(mean_absolute_error(self.y,self.y_pred),2)
        
        #root mean square percentage error
        self.rmspe = round(np.sqrt(np.sum(np.power(((self.y-self.y_pred)/self.y),2))/len(self.y)),3)
        
        #r2_score
        self.r2 = round(r2_score(self.y,self.y_pred),2)
        
        #adjusted_r2_score
        self.adjr2 = round(1-(1-r2_score(self.y,self.y_pred))*((self.x.shape[0]-1)/(self.x.shape[0]-self.x.shape[1]-1)),2)
        
        #dictionary storing all these testing score and this will be the returning value of function
        self.score_dict = {'Mean Square Error':self.mse, 'Root Mean Square Error':self.rmse,
                        'Mean Absolute Error':self.mae,'Root Mean Square Percentage Error':self.rmspe,'R2 Score':self.r2,
                        'Adjusted R2 Score': self.adjr2 }
        
        return self.score_dict

    def livelog(self, live):
        
        model_score = self.model_scores()
        live.log_param("Model", self.params['train_model']['model'])
        live.log_param("Model Hyperparametrs", self.params['train_model']['hyperparameters'])
        live.log_param('Features used in modelling', self.params['train_model']['features'])
        live.log_metric(f'{self.split} data/RMSE',model_score['Root Mean Square Error'])
        live.log_metric(f'{self.split} data/RMSPE',model_score['Root Mean Square Percentage Error'])
        live.log_metric(f'{self.split} data/MAE',model_score['Mean Absolute Error'])
        live.log_metric(f'{self.split} data/R2_SCORE',model_score['R2 Score'])
        fig = self.save_importance_plot()
        live.log_image(
            'Taxi Trip feature importances.png',fig
        )
        live.make_report()
   
    def save_importance_plot(self):
        """
        Save feature importance plot.

        Args:
            live (dvclive.Live): DVCLive instance.
            model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
            feature_names (list): List of feature names.
        """
        fig, axes = plt.subplots(dpi=100)
        axes.set_ylabel("Feature importance")

        importances = self.model.named_steps['regressor'].feature_importances_
        feature_names = self.model.named_steps['preprocess'].get_feature_names_out()
        forest_importances = pd.Series(importances, index=feature_names)
        forest_importances.plot.barh(ax=axes)

        return fig


        
@click.command()
@click.argument('train_input_filepath', type=click.Path())
@click.argument('test_input_filepath', type=click.Path())
@click.argument('model_path', type=click.Path())
def main(train_input_filepath,test_input_filepath, model_path):

    """ Runs data cleaning and splitting script to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    # Set up paths for input and output data
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    output_path = home_dir.as_posix() + '/dvclive'
    train_input_path = Path(data_dir.as_posix() + train_input_filepath)
    test_input_path = Path(data_dir.as_posix() + test_input_filepath)
    model = pickle.load(open(Path(home_dir.as_posix() + model_path), 'rb'))
    params_path = Path(home_dir.as_posix() + '/params.yaml')
    params = yaml.safe_load(open(params_path))
    
    df_train = pd.read_csv(train_input_path)
    x_train = df_train.drop(columns=['trip_duration'])
    y_train =df_train['trip_duration']

    df_test = pd.read_csv(test_input_path)
    x_test = df_test.drop(columns=['trip_duration'])
    y_test =df_test['trip_duration']


    train_model_score = ModelEvaluation( x_train, y_train, model,params,'train',output_path)
    test_model_score  = ModelEvaluation( x_test, y_test, model,params, 'test', output_path)

    with Live(output_path, save_dvc_exp=True) as live:
        train_model_score.livelog(live)
        test_model_score.livelog(live)
    
if __name__== "__main__":
    main()
    

