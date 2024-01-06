import boto3
from src.logger import infologger
from pathlib import Path 

class S3Push:

    def __init__(self):

         #trying to connect to s3
        try:
            self.s3 = boto3.client("s3")
            

        except Exception as e:

            #logger if connection with S3 fails
            infologger.info(f'Not able to Connect to S3 :  {e}')

    def push(self, file, bucket,name):

        try:
            self.s3.upload_file(file, bucket,name)

        except Exception as e:

            #logger if uploading fails
            infologger.info(f'Not able to upload to S3 :  {e}')


if __name__ == '__main__':
    
    path = Path(__file__)
    home_dir = path.parent
    mpath = Path(str(home_dir) + '/models/bestmodel.pkl')

    try:
        s3 = S3Push()
        s3.push(mpath, 'nyctrip-bucket', 'bestmodel.pkl')

    except Exception as e:
        infologger.info(f'Failed in pushing a model to S3 :  {e}')

    else:

        infologger.info('Model pushed to S3 successfully')

