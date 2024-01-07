import boto3

try:
            s3 = boto3.client("s3")

            s3.download_file(
                Bucket="nyctrip-bucket", Key="bestmodel.pkl", Filename="bestmodel.pkl"
            )

            s3.download_file(
                Bucket="nyctrip-bucket", Key="loc_kmeans.pkl", Filename="loc_kmeans.pkl"
            )

except Exception as e:

            #logger if connection with S3 fails
            print(f'Not able to Connect to S3 :  {e}')

else:

            #initialize parameters and path if connection and download of data from s3 is successful
            print('Successfully loaded')