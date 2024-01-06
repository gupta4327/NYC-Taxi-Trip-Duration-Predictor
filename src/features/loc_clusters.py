from sklearn.cluster import KMeans
import pickle
from src.logger import infologger
from pathlib import Path
import pandas as pd
import click
import yaml

class ClusteringLocation:
 
    def __init__(self, read_path):

         self.df = pd.read_csv(read_path)  


    def loc_cluster_creation(self, k, seed, home_dir):
        
        '''This function clusters the pickup and dropoff locations into K different clusters'''
        
        # Building kmeans cluster and clustering both pickup and dropoff into K different clusters
        pickup_coordinates = self.df[['pickup_latitude', 'pickup_longitude']].to_numpy()
        
        n_clusters = k
        
        try:
            # Clustering and labeling on pickup data
            loc_kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            loc_kmeans.fit(pickup_coordinates)

            # Dumping kmeans model
            pickupmodel = '/models/loc_kmeans.pkl'
            pickle.dump(loc_kmeans, open(Path(str(home_dir) + pickupmodel), 'wb'))
           
        except Exception as e:
            # Log if feature creation is errored out
            infologger.info(f'Location cluster failed with the error: {e}')
        
        else:
            # Log if created successfully
            infologger.info('Location cluster label feature created successfully')


#desigining main function
@click.command()
@click.argument('input_filepath', type=click.Path())
def main(input_filepath):

    # Set up paths for input, output and all other required ones
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    input_path = Path(data_dir.as_posix() + input_filepath)

    #loading parameters needed in script from params.yaml file 
    params_path = Path(home_dir.as_posix() + '/params.yaml')
    params = yaml.safe_load(open(params_path))['loc_clusters']

    loc_clustering = ClusteringLocation(input_path)
    loc_clustering.loc_cluster_creation(params['K'], params['seed'], home_dir)


if __name__ == '__main__':
    main()

