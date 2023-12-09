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
logger.info('Splitting into train test data from the whole data')

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
        try:
            # Read data from the provided CSV file
            self.df = pd.read_csv(self.read_path)
        except Exception as e:
            # Log if reading fails
            logger.info(f'Reading failed with error: {e}')
        else:
            # Log if reading is successful
            logger.info('Read performed successfully')

    def split_traintest(self):
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

# Command-line interface using Click
@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('split_percent', type=click.FLOAT)
@click.argument('seed', type=click.INT)
def main(input_filepath, output_filepath, split_percent, seed):
    """ Runs data splitting script to turn raw data from (../raw) into
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