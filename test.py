#importing necessary libraries
from src.features.build_features import BuildFeatures
import boto3  # pip install boto3
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from src.logger import infologger
import pickle
import yaml

#class to implement CI that will run predictor on test data and saves the scoring metrics graph

class CI_test:
    def __init__(self):
        
        #trying to connect to s3
        try:
            s3 = boto3.client("s3")


            s3.download_file(
                Bucket="nyctrip-bucket", Key="validate_data.csv", Filename="validate_data.csv"
            )

            s3.download_file(
                Bucket="nyctrip-bucket", Key="test_bestmodel.pkl", Filename="test_bestmodel.pkl"
            )

            s3.download_file(
                Bucket="nyctrip-bucket", Key="loc_kmeans.pkl", Filename="loc_kmeans.pkl"
            )

        except Exception as e:

            #logger if connection with S3 fails
            infologger.info(f'Not able to Connect to S3 :  {e}')

        else:

            #initialize parameters and path if connection and download of data from s3 is successful
            self.df = pd.read_csv("validate_data.csv")
            

    def predict(self):  

        #building a features for prediction on validate data 
        feat = BuildFeatures()
        self.df = feat.build(self.df, 'loc_kmeans.pkl') 
        features = yaml.safe_load(open('params.yaml'))['train_model']['features']
        self.df = self.df[features]

        #seperating input and output data 
        self.x = self.df.drop(columns=['trip_duration'])
        self.y = self.df['trip_duration']
        
        #initializing a object of predictor class
        predictor = pickle.load(open('test_bestmodel.pkl', 'rb'))
        
         #generating predictions
        self.y_pred = predictor.predict(self.x)
          
    def score(self):

        
        #root mean `square error
        rmse = round(np.sqrt(mean_squared_error(self.y,self.y_pred)),2)
        
        #mean absolute error
        mae = round(mean_absolute_error(self.y,self.y_pred),2)
        
        #root mean square percentage error
        rmspe = round(np.sqrt(np.sum(np.power(((self.y-self.y_pred)/self.y),2))/len(self.y))*100,3)
        
        #r2_score
        r2 = round(r2_score(self.y,self.y_pred)*100,2)
        
        
        #dictionarself.y storing all these testing score and this will be the returning value of function
        self.score_dict = { 'Root Mean Square Error':rmse,
                        'Mean Absolute Error':mae,'Root Mean Square Percentage Error':rmspe,'R2 Score':r2
                        }
        
        return self.score_dict

    def test(self):

        try:
            self.predict()
            self.score()
            fig, ax = plt.subplots()

            ax.bar(list(self.score_dict.keys()), list(self.score_dict.values()))

            ax.set_ylabel('Score')
            ax.set_xlabel('Metrices')
            ax.set_title('Different Scoring Metrices for model')
            plt.xticks(rotation = 'vertical')
            plt.savefig('metrices_bars.png')


        except Exception as e:
            self.message = 'Error in plotting and predicting : ' + str(e)
            infologger.info(self.message)

        else:
            self.message = 'Plotted successful'
            infologger.info(self.message)

if __name__ == '__main__':
    tst = CI_test()
    tst.test()    

