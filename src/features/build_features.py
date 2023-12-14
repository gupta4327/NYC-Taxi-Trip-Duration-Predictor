# Importing necessary libraries and modules
import pandas as pd
import numpy as np
import yaml
import click
import logging
from sklearn.cluster import KMeans
import haversine as hs
from haversine import Unit
from pathlib import Path
import pickle

# Setting up logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('build_features.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('Creating new useful features out of the existing features')

# Class for building features from dataset
class BuildFeatures:
    def __init__(self, read_path, k=None, seed=None, write_path=None, home_dir=None):
        self.read_path = read_path
        self.write_path = write_path
        self.k = k
        self.seed = seed
        self.home_dir = home_dir

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

    def date_related_features(self):
        '''This function creates features such as day, hour, weekday from pickup date'''
        try:
            # Converting pickup date and dropoff date into datetime objects
            self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
            self.df['dropoff_datetime'] = pd.to_datetime(self.df['dropoff_datetime'])

            # Extracting year, month, day, week of day, and hours from datetime
            self.df['pickup_day'] = self.df['pickup_datetime'].dt.day
            self.df['pickup_weekday'] = self.df['pickup_datetime'].dt.dayofweek
            self.df['pickup_hour'] = self.df['pickup_datetime'].dt.hour
            weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            self.df['pickup_weekday'] = self.df['pickup_weekday'].map(lambda x: weekday[x])
        except Exception as e:
            # Log if feature creation from pickup date fails
            logger.info(f'Feature extraction from pickup date has failed with the error: {e}')
        else:
            # Log if feature extraction from date has been successful
            logger.info('Feature extraction from pickup date has been completed successfully')

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
            logger.info(f'Dayphase feature creation failed with the error: {e}')
        else:
            # Log if created successfully
            logger.info('Day Phase feature created successfully')

    def loc_cluster_creation(self):
        '''This function clusters the pickup and dropoff locations into K different clusters'''
        # Building kmeans cluster and clustering both pickup and dropoff into K different clusters
        pickup_coordinates = self.df[['pickup_latitude', 'pickup_longitude']]
        dropoff_coordinates = self.df[['dropoff_latitude', 'dropoff_longitude']]
        n_clusters = self.k
        try:
            # Clustering and labeling on pickup data
            pickup_kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
            self.df['pickup_cluster_label'] = pickup_kmeans.fit_predict(pickup_coordinates)

            # Clustering and labeling on dropoff data
            dropoff_kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
            self.df['dropoff_cluster_label'] = dropoff_kmeans.fit_predict(dropoff_coordinates)

            # Dumping kmeans model
            pickupmodel = '/models/pickup_kmeans.pkl'
            pickle.dump(pickup_kmeans, open(Path(str(self.home_dir) + pickupmodel), 'wb'))
            dropoffmodel = '/models/dropoff_kmeans.pkl'
            pickle.dump(dropoff_kmeans, open(Path(str(self.home_dir) + dropoffmodel), 'wb'))
        except Exception as e:
            # Log if feature creation is errored out
            logger.info(f'Dropoff or Pickup cluster feature creation failed with the error: {e}')
        else:
            # Log if created successfully
            logger.info('Pickup and Dropoff cluster label feature created successfully')

    def cluster_assign(self):
        # Building kmeans cluster and clustering both pickup and dropoff into K different clusters
        pickup_coordinates = self.df[['pickup_latitude', 'pickup_longitude']]
        dropoff_coordinates = self.df[['dropoff_latitude', 'dropoff_longitude']]
        pickupmodel = '/models/pickup_kmeans.pkl'
        pickup_kmeans = pickle.load(open(Path(str(self.home_dir) + pickupmodel), 'rb'))
        dropoffmodel = '/models/dropoff_kmeans.pkl'
        dropoff_kmeans = pickle.load(open(Path(str(self.home_dir) + dropoffmodel), 'rb'))
        self.df['pickup_cluster_label'] = pickup_kmeans.predict(pickup_coordinates)
        self.df['dropoff_cluster_label'] = dropoff_kmeans.predict(dropoff_coordinates)

    # Function to calculate distance from latitude and longitude using haversine
    def distance_calculator(self, x):
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
            logger.info(f'Distance feature creation failed with the error: {e}')
        else:
            # Log if created successfully
            logger.info('Distance feature created successfully')

    def write_data(self):
        '''This function writes the data into the destination folder'''
        try:
            # Write the training and testing sets to CSV files
            filename = str(self.read_path)
            filename = str(filename.encode('unicode_escape')).split("\\")[-1]
            l = len(filename)
            filename = filename[0:l - 1]
            filename = '/' + filename
            self.df.to_csv(Path(str(self.write_path) + filename), index=False)
        except Exception as e:
            # Log if writing fails
            logger.info(f'Writing data failed with error: {e}')
        else:
            # Log if writing is successful
            logger.info('Write performed successfully')

    def fit(self):
        '''This function needs to be run for training data'''
        self.read_data()
        self.date_related_features()
        self.dayphase_feature()
        self.distance_feature()
        self.loc_cluster_creation()
        if self.write_path is not None:
            self.write_data()
        return self.df

    def transform(self):
        '''This function needs to be run on test and predicting data'''
        self.read_data()
        self.date_related_features()
        self.dayphase_feature()
        self.distance_feature()
        self.cluster_assign()
        if self.write_path is not None:
            self.write_data()
        return self.df


@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs feature building script to turn train and test data from given input path
        (default is ../interim) into new data with added features and stores it into given
        output path(default is ../processed).
    """
    # Set up paths for input, output and all other required ones
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    input_path = Path(data_dir.as_posix() + input_filepath)
    if output_filepath != str(None):
        output_path = Path(data_dir.as_posix() + output_filepath)
    else:
        output_path = None
    params_path = Path(home_dir.as_posix() + '/params.yaml')
    params = yaml.safe_load(open(params_path))['build_features']

    feat = BuildFeatures(input_path, params['K'], params['seed'], output_path, home_dir)
    if 'train' in input_filepath:
        feat.fit()
    else:
        feat.transform()


if __name__ == "__main__":
    main()
