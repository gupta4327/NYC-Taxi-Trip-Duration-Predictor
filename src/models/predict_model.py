import pandas as pd
import pickle
from src.features.build_features import BuildFeatures
from pathlib import Path
import yaml

class TripDurationPredictor:

    def __init__(self):
        self.curr_dir = Path(__file__)
        self.home_dir = self.curr_dir.parent.parent.parent
        self.model_path = Path(str(self.home_dir) + '/models/bestmodel.pkl')
        self.params_path = Path(str(self.home_dir) + '/params.yaml')
        self.features = yaml.safe_load(open(self.params_path))['train_model']['features']
        self.features.remove('trip_duration')

    def dict_to_df(self,dict):
        
        return pd.DataFrame(dict, index=[0])

    def loadmodel(self):
        self.model = pickle.load(open(self.model_path, 'rb'))

    def buildfeatures(self):
        feat = BuildFeatures()
        self.df = feat.build(self.df)
        self.df = self.df[self.features]
        
    def model_load(self):
        self.model = pickle.load(open(self.model_path, 'rb'))

    def predict_duration(self,dict):

        self.df = self.dict_to_df(dict)
        self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
        self.buildfeatures()
        self.model_load()
        pred = self.model.predict(self.df)
        return pred[0]
    
if __name__ == "__main__":

        t = TripDurationPredictor()
        dict = {'vendor_id':[2], 'pickup_latitude':[40.767936706542969], 'pickup_longitude':[-73.982154846191406],
                'dropoff_latitude':[40.765602111816406], 'dropoff_longitude':[-73.964630126953125], 
                'pickup_datetime':['2016-03-14 17:24:55'], 'store_and_fwd_flag':['N']}
        print(t.predict_duration(dict)[0])
        