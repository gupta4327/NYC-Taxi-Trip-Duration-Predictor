from src.models.predict_model import TripDurationPredictor
from src.models.train_model import model_eval
import boto3  # pip install boto3
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

class CI_test:
    def __init__(self):
        try:
            s3 = boto3.client("s3")


            s3.download_file(
                Bucket="nyctrip-bucket", Key="test_data.csv", Filename="data/interim/test_data.csv"
            )

            s3.download_file(
                Bucket="nyctrip-bucket", Key="bestmodel.pkl", Filename="models/bestmodel.pkl"
            )

        except Exception as e:
            print(e)

        else:
            self.read_path = Path(r'C:\Users\Aman Gupta\test\nyc_taxi_trip_duration_predictor\data\interim\test_data.csv')
            self.model_path = Path(r'C:\Users\Aman Gupta\test\nyc_taxi_trip_duration_predictor\models\bestmodel.pkl')
            self.df = pd.read_csv(self.read_path)
            self.x = self.df.drop(columns=['trip_duration'])
            self.y = self.df['trip_duration']
            self.y_pred =[]

    def predict(self):    
        predictor = TripDurationPredictor()
        listofdict = self.x.to_dict("records")
        for dct in listofdict:
            self.y_pred.append(predictor.predict_duration(dct))

    def score(self):

        #mean square error
        mse = round(mean_squared_error(self.y,self.y_pred),2)
            
        #root mean `square error
        rmse = round(np.sqrt(mean_squared_error(self.y,self.y_pred)),2)
        
        #mean absolute error
        mae = round(mean_absolute_error(self.y,self.y_pred),2)
        
        #root mean square percentage error
        rmspe = round(np.sqrt(np.sum(np.power(((self.y-self.y_pred)/self.y),2))/len(self.y)),3)
        
        #r2_score
        r2 = round(r2_score(self.y,self.y_pred),2)
        
        #adjusted_r2_score
        adjr2 = round(1-(1-r2_score(self.y,self.y_pred))*((self.x.shape[0]-1)/(self.x.shape[0]-self.x.shape[1]-1)),2)
        
        #dictionarself.y storing all these testing score and this will be the returning value of function
        self.score_dict = {'Mean Square Error':mse, 'Root Mean Square Error':rmse,
                        'Mean Absolute Error':mae,'Root Mean Square Percentage Error':rmspe,'R2 Score':r2,
                        'Adjusted R2 Score': adjr2 }
        
        return self.score_dict

    def test(self):

        try:
            self.predict()
            self.score()
            fig, ax = plt.subplots()

            ax.bar(list(self.score_dict.keys()), list(self.score_dict.keys()))

            ax.set_ylabel('Score')
            ax.set_xlabel('Metrices')
            ax.set_title('Different Scoring Metrices for model')
           
            plt.savefig('metrices_bars.png', bbox_inches='tight')


        except Exception as e:
            self.message = 'Error in plotting and predicting : ' + e

        else:
            self.message = 'Plotted successful'

if __name__ == '__main__':
    tst = CI_test()
    tst.test()    

