import boto3

def lambda_exists(lambda_name):
    lambda_client =boto3.client('lambda')

def s3bucket_exists(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    if bucket.creation_date:
        return True
    else:
        return False