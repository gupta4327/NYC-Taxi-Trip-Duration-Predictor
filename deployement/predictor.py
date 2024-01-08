#importing necessary libraries
import pandas as pd
import pickle
from build_features import BuildFeatures
import yaml
from pathlib import Path

#class designed to do production predictions
class TripDurationPredictor:

    def __init__(self):

        #initialising all necessary paths and parameters 
        dir_path = Path(__file__).parent
        self.cluster = Path(dir_path.as_posix() + '/loc_kmeans.pkl')
        self.model_path = Path(dir_path.as_posix()+'/bestmodel.pkl')
        self.params_path =Path(dir_path.as_posix()+ '/features.yaml')
        self.features = yaml.safe_load(open(self.params_path))['model']['features']

    def dict_to_df(self,dict):
        
        #converting recieved input dictionary to dataframe
        return pd.DataFrame(dict,index =[0])

   
    def buildfeatures(self):

        #building necessary features required for prediction
        feat = BuildFeatures()
        self.df = feat.build(self.df, self.cluster)
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
             print(f'Prediction has been failed because of error : {e}')
        else:
            return pred
    
if __name__ == "__main__":

        #dummy example if the main file has been ran directly 
        t = TripDurationPredictor()
        dict = {'vendor_id':[2], 'pickup_latitude':[40.767936706542969], 'pickup_longitude':[-73.982154846191406],
                'dropoff_latitude':[40.765602111816406], 'dropoff_longitude':[-73.964630126953125], 
                'pickup_datetime':['2016-03-14 17:24:55'], 'store_and_fwd_flag':['N']}
        print(t.predict_duration(dict)[0])
        