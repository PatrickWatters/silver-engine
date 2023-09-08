import boto3
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import logging
import os
import json
import base64
import redis
from boto3.s3.transfer import TransferConfig

# ===================================== Setings ==================================================================
s3 = boto3.client('s3')
redis_client = None
GB = 1024 ** 3
config = TransferConfig(use_threads=True,max_concurrency=10,multipart_threshold=5*GB) # To consume less downstream bandwidth, decrease the maximum concurrency
# ===================================== Setings ==================================================================
logger = logging.getLogger()

def lambda_handler(event, context):
    # Set the log level based on a variable configured in the Lambda environment.
    logger.setLevel(os.environ.get('LOG_LEVEL', logging.INFO))
    logger.debug('Event: %s', event)
    batch_metadata = event['batch_metadata']
    batch_id = event['batch_id']
    cache_bacth= event['cache_bacth'] 
    return_batch_data= event['return_batch_data'] 
    bucket_name=event['bucket']
    cache_error_message = ''
    if cache_bacth:
        redis_client = redis.StrictRedis(host=event['redis_host'], port=event['redis_port'])
    try:
        #maybe do some check to check Redis is working... 
        #return {'batch_id': redis_client.dbsize(),'isCached': redis_client.dbsize()}
        samples = []
        for key, result in fetch_data_from_s3(bucket_name=bucket_name,batch_metadata=batch_metadata,func_name=load_img_as_base64):
            samples.append(result)
        json_samples = json.dumps(samples) 
        isCached = False

        if cache_bacth:      
            try:
                redis_client.set(batch_id, json_samples)
                isCached = True
            except Exception as e:
                isCached = False
                cache_error_message = str(e)  
                logger.error(e, exc_info=True)

        if not isCached or return_batch_data:
                return {'batch_id': batch_id,'isCached': isCached,'batch_data':json_samples,'cache_error_message':cache_error_message}
        else:
            return {'batch_id': batch_id,'isCached': isCached,'cache_error_message':cache_error_message}
        
    except Exception as e:  # Catch all for easier error tracing in logs
        logger.error(e, exc_info=True)
        emesage = 'Error occurred during execution:' + str(e)
        raise Exception(emesage)  # notify aws of failure


def fetch_data_from_s3(bucket_name,batch_metadata,func_name):
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_key = {executor.submit(func_name, key,bucket_name): key for key in batch_metadata}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()
            if not exception:
                yield key, future.result()
            else:
                yield key, exception

def load_img_as_base64(bacth_item, s3_bucket):
    file_path = bacth_item[0]
    label = bacth_item[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        base_64_encoded_img = base64.b64encode(f.getvalue()).decode('utf-8')
    return base_64_encoded_img, label


'''
if __name__ == "__main__":
    import random
    #create some dummy data
    batch_metadata = []
    for i in range(0,10):
        batch_metadata.append(('test/Frog/leopard_frog_s_001876.png',6))
    #set configs for the command
    action_params = {}
    action_params['batch_metadata'] = batch_metadata
    action_params['batch_id'] = random.randint(0,10000000000)
    action_params['cache_bacth'] = True
    action_params['return_batch_data'] = False
    action_params['bucket'] = 'sdl-cifar10'    
    action_params['redis_host'] = '127.0.0.1'    
    action_params['redis_port'] = 6379

    #call function
    response_message = lambda_handler(action_params, None)
    print(response_message)
'''