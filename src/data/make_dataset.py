# Import necessary libraries and modules
import click
import logging
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.logger import infologger

# Log information about the script starting
infologger.info('Basic cleaning and Splitting into train test data from the whole data')

# Class for creating train and test datasets
class TrainTestCreation:

    def __init__(self, read_path, params, write_path=None):
        # Initialize class variables with provided parameters
        self.read_path = read_path 
        self.write_path = write_path 
        self.test_per = params['test_per']
        self.seed = params['seed']
        self.trip_duration_lowlimit = params['trip_duration_lowlimit']
        self.trip_duration_uplimit = params['trip_duration_uplimit']
        self.pickup_latitude_lowlimit = params['pickup_latitude_lowlimit']
        self.pickup_latitude_uplimit = params['pickup_latitude_uplimit']
        self.dropoff_latitude_lowlimit = params['dropoff_latitude_lowlimit']
        self.dropoff_latitude_uplimit = params['dropoff_latitude_uplimit']
        self.pickup_longitude_lowlimit = params['pickup_longitude_lowlimit']
        self.pickup_longitude_uplimit = params['pickup_longitude_uplimit']
        self.dropoff_longitude_lowlimit = params['dropoff_longitude_lowlimit']
        self.dropoff_longitude_uplimit = params['dropoff_longitude_uplimit']

        # Log information about the parameters passed to the class
        infologger.info(f'Call to make_dataset with the parameters: Data Read Path: {self.read_path}, Data write path: {self.write_path}, Test Percentage: {self.test_per}, and seed value: {self.seed}')
        
    def read_data(self):
        '''This function reads data from input path and stores it into a dataframe'''
        try:
            # Read data from the provided CSV file
            self.df = pd.read_csv(self.read_path)
        except Exception as e:
            # Log if reading fails
            infologger.info(f'Reading failed with error: {e}')
        else:
            # Log if reading is successful
            infologger.info('Read performed successfully')

    def date_type_conversion(self):
        '''This function reads all the date columns in data from object datatype to datetime datatype'''

        try:
            #converting pickup timestamp into datetime object
            self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
            
            #converting dropoff timestamp into datetime object
            self.df['dropoff_datetime'] = pd.to_datetime(self.df['dropoff_datetime'])

        except Exception as e:
            # Log if object into date conversion fails
            infologger.info(f'Date conversion of columns has failed with error : {e}')

        else:
            # Log if object into datetime conversion is successful
            infologger.info('Date conversion performed successfully')


    def outlier_removal(self):
        '''This function removes the outlier from the data based on upper and lower threshold provided'''

        try:
            self.df = self.df[(self.df['trip_duration']>=10) & (self.df['trip_duration']<=30000)]
            self.df = self.df.loc[(self.df['pickup_latitude'] >= 40.637044) & (self.df['pickup_latitude'] <= 40.855256)]
            self.df = self.df.loc[(self.df['pickup_longitude'] >= -74.035735) & (self.df['pickup_longitude'] <= -73.770272)]
            self.df = self.df.loc[(self.df['dropoff_latitude'] >= 40.637044) & (self.df['dropoff_latitude'] <= 40.855256)]
            self.df = self.df.loc[(self.df['dropoff_longitude'] >= -74.035735) & (self.df['dropoff_longitude'] <= -73.770272)]

        except Exception as e:
            #log if outlier removal is failed
            infologger.info(f'Outlier removal for data has failed with error : {e}')

        else:
            #log if outlier removal is successful
            infologger.info(f'Outlier removal performed successfully')

    def split_traintest(self):

        '''This function splits the whole data into train test as per the test percent provided'''

        try:
            # Split the data into training and testing sets
            self.train_data, self.test_data = train_test_split(self.df, random_state=self.seed,test_size=self.test_per)
        except Exception as e:
            # Log if splitting fails
            infologger.info(f'Splitting failed with error: {e}')
        else:
            # Log if splitting is successful
            infologger.info('Split performed successfully')

    def write_data(self):

        '''This function writes the data into destination folder'''
        try:
            # Write the training and testing sets to CSV files
            self.train_data.to_csv(Path(str(self.write_path) + '/train_data.csv'),index=False)
            self.test_data.to_csv(Path(str(self.write_path) + '/test_data.csv'),index=False)
        except Exception as e:
            # Log if writing fails
            infologger.info(f'Writing data failed with error: {e}')
        else:
            # Log if writing is successful
            infologger.info('Write performed successfully')

    def fit(self):
        self.read_data()
        self.date_type_conversion()
        self.outlier_removal()
        self.split_traintest()
        if self.write_path !=None:
            self.write_data()

        return (self.train_data, self.test_data)

# Command-line interface using Click
@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):

    """ Runs data cleaning and splitting script to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    # Set up paths for input and output data
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    input_path = Path(data_dir.as_posix() + input_filepath)
    output_path = Path(data_dir.as_posix() + output_filepath)
    params_path = Path(home_dir.as_posix()+'/params.yaml')
    params=yaml.safe_load(open(params_path))['make_dataset']
    
    # Create an instance of the train_test_creation class
    if output_filepath != 'None':
        split_data = TrainTestCreation(input_path, params, output_path)

    else:
        split_data = TrainTestCreation(input_path, params)
    
    # Perform the steps of reading, splitting, and writing data
    split_data.fit()

# Execute the main function if the script is run
if __name__ == '__main__':
    main()