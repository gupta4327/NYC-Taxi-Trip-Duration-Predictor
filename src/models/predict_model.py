#importing necessary libraries
import pandas as pd
import pickle
from src.features.build_features import BuildFeatures
from pathlib import Path
import yaml
from src.logger import infologger

#class designed to do production predictions
class TripDurationPredictor:

    def __init__(self):

        #initialising all necessary paths and parameters 
        self.curr_dir = Path(__file__)
        self.home_dir = self.curr_dir.parent.parent.parent
        self.model_path = Path(str(self.home_dir) + '/models/bestmodel.pkl')
        self.params_path = Path(str(self.home_dir) + '/params.yaml')
        self.features = yaml.safe_load(open(self.params_path))['train_model']['features']
        self.features.remove('trip_duration')

    def dict_to_df(self,dict):
        
        #converting recieved input dictionary to dataframe
        return pd.DataFrame(dict)

   
    def buildfeatures(self):

        #building necessary features required for prediction
        feat = BuildFeatures()
        self.df = feat.build(self.df)
        self.df = self.df[self.features]
        
    def model_load(self):

        #loading a binary model through pickle
        self.model = pickle.load(open(self.model_path, 'rb'))

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
        