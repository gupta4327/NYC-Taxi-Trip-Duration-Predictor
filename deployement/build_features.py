# Importing necessary libraries and modules
import pandas as pd
import click
import haversine as hs
from haversine import Unit
from pathlib import Path
import pickle
from logger import infologger

#displaying info on logfile
infologger.info('Creating new useful features out of the existing features')

# Class for building features from dataset
class BuildFeatures:

    def read_data(self,read_path):
        '''This function reads csv data from input path and stores it into a dataframe'''
        
        try:
            # Read data from the provided CSV file
            self.df = pd.read_csv(read_path)
        
        except Exception as e:    
            # Log if reading fails
            infologger.info(f'Reading failed with error: {e}')
        
        else:
            # Log if reading is successful
            infologger.info('Read performed successfully')

    def date_related_features(self):
        '''This function creates features such as day, hour, weekday from pickup date'''
        try:

            # Converting pickup date and dropoff date into datetime objects
            self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
            
            # Extracting year, month, day, week of day, and hours from datetime
            self.df['pickup_day'] = self.df['pickup_datetime'].dt.day
            self.df['pickup_weekday'] = self.df['pickup_datetime'].dt.dayofweek
            self.df['pickup_hour'] = self.df['pickup_datetime'].dt.hour
            weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            self.df['pickup_weekday'] = self.df['pickup_weekday'].map(lambda x: weekday[x])
        
        except Exception as e:
            # Log if feature creation from pickup date fails
            infologger.info(f'Feature extraction from pickup date has failed with the error: {e}')
        
        else:
            # Log if feature extraction from date has been successful
            infologger.info('Feature extraction from pickup date has been completed successfully')

    

    def trip_dayphase(self, x):
        
        '''Function to categorize complete day into different phases'''
        
        if 0 <= x < 6:
            return 'overnight'
        if 6 <= x <= 12:
            return 'morning'
        if 12 < x <= 17:
            return 'afternoon'
        
        else:
            return 'evening/night'

    def dayphase_feature(self):
        
        '''This function uses trip_dayphase function and creates a day phase feature'''
        
        try:
            # Applying trip day phase function
            self.df['day_phase'] = self.df['pickup_hour'].apply(lambda x: self.trip_dayphase(x))
        
        except Exception as e:
            # Log if feature creation is errored out
            infologger.info(f'Dayphase feature creation failed with the error: {e}')
        
        else:
            # Log if created successfully
            infologger.info('Day Phase feature created successfully')


    def cluster_assign(self,loc_model):
        
        # assiging a pickup and dropoff geopositions a clusters
        pickup_coordinates = self.df[['pickup_latitude', 'pickup_longitude']].to_numpy()
        dropoff_coordinates = self.df[['dropoff_latitude', 'dropoff_longitude']].to_numpy()
        
        loc_kmeans = pickle.load(open(Path(str(loc_model)), 'rb'))
        
        #assiging cluster to each geolocation
        self.df['pickup_cluster_label'] = loc_kmeans.predict(pickup_coordinates)
        self.df['dropoff_cluster_label'] = loc_kmeans.predict(dropoff_coordinates)

    
    def distance_calculator(self, x):
        
        '''This Function to calculate distance from latitude and longitude using haversine'''
        
        loc1 = (x['pickup_latitude'], x['pickup_longitude'])
        loc2 = (x['dropoff_latitude'], x['dropoff_longitude'])
        distance = hs.haversine(loc1, loc2, unit=Unit.METERS)
        return distance

    def distance_feature(self):
        '''This function uses distance calculator function to calculate distance between trips'''
        try:
            # Applying a distance function
            self.df['trip_distance'] = self.df.apply(lambda x: self.distance_calculator(x), axis=1)
        
        except Exception as e:
            # Log if feature creation is errored out
            infologger.info(f'Distance feature creation failed with the error: {e}')
        
        else:
            # Log if created successfully
            infologger.info('Distance feature created successfully')

    def write_data(self,read_path,write_path):
        
        '''This function writes the data into the destination folder'''
        
        try:
            # Write the training and testing sets to CSV files
            filename = str(read_path)
            filename = str(filename.encode('unicode_escape')).split("\\")[-1]
            l = len(filename)
            filename = filename[0:l - 1]
            filename = '/' + filename
            self.df.to_csv(Path(str(write_path) + filename), index=False)
        
        except Exception as e:
            # Log if writing fails
            infologger.info(f'Writing data failed with error: {e}')
        
        else:
            # Log if writing is successful
            infologger.info('Write performed successfully')



    def fit(self,read_path,write_path,locmodel):
        
        '''This function needs to be run on data as it runs all the functions sequentially to perform an desired action'''
        
        self.read_data(read_path)
        self.date_related_features()
        self.dayphase_feature()
        self.distance_feature()
        self.cluster_assign(locmodel)
        self.write_data(read_path, write_path)

    def build(self,df,locmodel):
        self.df = df
        self.date_related_features()
        self.dayphase_feature()
        self.distance_feature()
        self.cluster_assign(locmodel)
        return self.df


#desigining main function
@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('modelpath', type= click.Path())
def main(input_filepath, output_filepath, model_filepath):
    
    """ Runs feature building script to turn train and test data from given input path
        (default is ../interim) into new data with added features and stores it into given
        output path(default is ../processed).
    """
    # Set up paths for input, output and all other required ones
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    input_path = Path(data_dir.as_posix() + input_filepath)
    output_path = Path(data_dir.as_posix() + output_filepath)
    locmodel_path = Path(data_dir.as_posix() + model_filepath)

    #initiating a class object 
    feat = BuildFeatures()
   
    #fiting function to transform data 
    feat.fit(input_path, output_path, locmodel_path)
    

if __name__ == "__main__":
    main()
