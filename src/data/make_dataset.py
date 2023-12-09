# Import necessary libraries and modules
import click
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('make_dataset.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Log information about the script starting
logger.info('Basic cleaning and Splitting into train test data from the whole data')

# Class for creating train and test datasets
class train_test_creation:

    def __init__(self, read_path, write_path, test_per, seed):
        # Initialize class variables with provided parameters
        self.read_path = read_path 
        self.write_path = write_path 
        self.test_per = test_per
        self.seed = seed

        # Log information about the parameters passed to the class
        logger.info(f'Call to make_dataset with the parameters: Data Read Path: {self.read_path}, Data write path: {self.write_path}, Test Percentage: {self.test_per}, and seed value: {self.seed}')
        
    def read_data(self, path):
        '''This function reads data from input path and stores it into a dataframe'''
        try:
            # Read data from the provided CSV file
            self.df = pd.read_csv(self.read_path)
        except Exception as e:
            # Log if reading fails
            logger.info(f'Reading failed with error: {e}')
        else:
            # Log if reading is successful
            logger.info('Read performed successfully')

    def date_type_conversion(self):
        '''This function reads all the date columns in data from object datatype to datetime datatype'''

        try:
            #converting pickup timestamp into datetime object
            self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
            
            #converting dropoff timestamp into datetime object
            self.df['dropoff_datetime'] = pd.to_datetime(self.df['dropoff_datetime'])

        except Exception as e:
            # Log if object into date conversion fails
            logger.info(f'Date conversion of columns has failed with error : {e}')

        else:
            # Log if object into datetime conversion is successful
            logger.info('Date conversion performed successfully')


    def outlier_removal(self):
        '''This function removes the outlier from the data based on upper and lower threshold provided'''

        try:
            self.df = self.df[(self.df['trip_duration']>=self.upper_threshold) & (self.df['trip_duration']<=self.lower_threshold)]
            self.df = self.df.loc[(self.df['pickup_latitude'] >= 40.637044) & (self.df['pickup_latitude'] <= 40.855256)]
            self.df = self.df.loc[(self.df['pickup_longitude'] >= -74.035735) & (self.df['pickup_longitude'] <= -73.770272)]
            self.df = self.df.loc[(self.df['dropoff_latitude'] >= 40.637044) & (self.df['dropoff_latitude'] <= 40.855256)]
            self.df = self.df.loc[(self.df['dropoff_longitude'] >= -74.035735) & (self.df['dropoff_longitude'] <= -73.770272)]

        except Exception as e:
            #log if outlier removal is failed
            logger.info(f'Outlier removal for data has failed with error : {e}')

        else:
            #log if outlier removal is successful
            logger.info(f'Outlier removal performed successfully')

    def split_traintest(self):

        '''This function splits the whole data into train test as per the test percent provided'''

        try:
            # Split the data into training and testing sets
            self.train_data, self.test_data = train_test_split(self.df, test_per=self.test_per, random_state=self.seed)
        except Exception as e:
            # Log if splitting fails
            logger.info(f'Splitting failed with error: {e}')
        else:
            # Log if splitting is successful
            logger.info('Split performed successfully')

    def write_data(self):

        '''This function writes the data into destination folder'''

        try:
            # Write the training and testing sets to CSV files
            self.train_data.to_csv(self.write_path + '/train_data.csv')
            self.test_data.to_csv(self.write_path + '/test_data.csv')
        except Exception as e:
            # Log if writing fails
            logger.info(f'Writing data failed with error: {e}')
        else:
            # Log if writing is successful
            logger.info('Write performed successfully')

    def fit(self):
        self.read_data()
        self.date_type_conversion()
        self.outlier_removal()
        self.write_data()

# Command-line interface using Click
@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('split_percent', type=click.FLOAT)
@click.argument('seed', type=click.INT)
def main(input_filepath, output_filepath, split_percent, seed):
    """ Runs data cleaning and splitting script to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    # Set up paths for input and output data
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    input_path = Path(data_dir.as_posix() + input_filepath)
    output_path = Path(data_dir.as_posix() + output_filepath)

    # Create an instance of the train_test_creation class
    split_data = train_test_creation(input_path, output_path, split_percent, seed)
    
    # Perform the steps of reading, splitting, and writing data
    split_data.read_data()
    split_data.split_traintest()
    split_data.write_data()

# Execute the main function if the script is run
if __name__ == '__main__':
    main()