#importing necessary libraries
import pandas as pd
import pickle
from build_features import BuildFeatures
import yaml
from pathlib import Path
from logger import infologger

#class designed to do production predictions
class TripDurationPredictor:

    def __init__(self):

        #initialising all necessary paths and parameters 
        try:
             
            dir_path = Path(__file__).parent
            self.cluster = Path(dir_path.as_posix() + '/loc_kmeans.pkl')
            self.model_path = Path(dir_path.as_posix()+'/bestmodel.pkl')
            self.params_path =Path(dir_path.as_posix()+ '/features.yaml')
            self.features = yaml.safe_load(open(self.params_path))['model']['features']

        except Exception as e:

            infologger.info(f'Initialization has been failed with error : {e}')

        else:
            infologger.info('Initialization has been done successfully')

    
    def dict_to_df(self,dict):
        
        #converting recieved input dictionary to dataframe
        try:

            return pd.DataFrame(dict,index =[0])
        
        except Exception as e:

            infologger.info(f'Dataframe creation from input dictionary has been failed with error : {e}')           

   
    def buildfeatures(self):

        #building necessary features required for prediction
        try:
            feat = BuildFeatures()
            self.df = feat.build(self.df, self.cluster)
            self.df = self.df[self.features]
        
        except Exception as e:
            infologger.info(f'Feature build has been failed with error : {e}')

        else:
            infologger.info('Features created successfully')

    def model_load(self):

        try:
            #loading a binary model through pickle
            self.model = pickle.load(open(self.model_path, 'rb'))
        
        except Exception as e:

            infologger.info(f'Model loading has been failed with error : {e}')

        else:
            infologger.info('Model loaded successfully')

    def predict_duration(self,dict):

          # function that performs all other function together and returns a predicted output
        try:
            self.df = self.dict_to_df(dict)
            self.buildfeatures()
            self.model_load()
            pred = self.model.predict(self.df)
        
        except Exception as e:
             infologger.info(f'Prediction has been failed because of error : {e}')
        
        else:
            return pred
    
if __name__ == "__main__":

        #dummy example if the main file has been ran directly 
        t = TripDurationPredictor()
        dict = {'vendor_id':[2], 'pickup_latitude':[40.767936706542969], 'pickup_longitude':[-73.982154846191406],
                'dropoff_latitude':[40.765602111816406], 'dropoff_longitude':[-73.964630126953125], 
                'pickup_datetime':['2016-03-14 17:24:55'], 'store_and_fwd_flag':['N']}
        print(t.predict_duration(dict)[0])
        