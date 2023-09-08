import grpc
from concurrent import futures
import cms_pb2_grpc as pb2_grpc
import cms_pb2 as pb2
from configparser import ConfigParser
import os.path
import misc.aws as aws
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
from dataset import Dataset
from job import MLTrainingJob
import logging
import json
from batch import BatchGroup
from unique_priority_queue import UniquePriorityQueue
from lambda_wrapper import LambdaWrapper
from misc.redis_client import RedisClient

class CacheManagementService(pb2_grpc.CacheManagementServiceServicer):

    def __init__(self, *args, **kwargs):
        print(f'Running Setup for Cache Management Service')
        self.batch_groups:Dict[str, BatchGroup] = {}
        self.active_training_jobs = {}
        self.global_batch_group_idx = 0
        self.global_queue = UniquePriorityQueue()
        self.global_queue.batch_groups = self.batch_groups
        self.global_queue.start_consumers(num_consumers=0)
        self._read_config()
        self._check_environment()
        self.lambda_wrapper = LambdaWrapper(self.bucket_name, self.redis_host, self.redis_port,function_name=self.lambda_func_name)
        self.redis_client:RedisClient = RedisClient(self.redis_host,self.redis_port)
        self._gen_new_group_of_batches() #create an initial set of batches

        #queue up the first few batches for the first job first epoch
        '''
        firstset =  self.batch_groups[self.global_batch_group_idx].get_batch_ids()
        for i in range(0, self.warm_up_distance):
            batch_id = firstset[i]
            self.global_queue.put((100,(1000, self.global_batch_group_idx, batch_id, 'prefetch')))
        '''
        pass


    def _read_config(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_filepath = dir_path+"/config.ini"
        # check if the config file exists
        if os.path.exists(config_filepath):
            config = ConfigParser()
            config.read(config_filepath)
            self.micro_batch_size = int(config["SUPER"]["micro_batch_size"])
            self.bucket_name = config["S3"]["bucket"]
            self.use_substitutional_hits = config.getboolean('SUPER','use_substitutional_hits')
            self.drop_last =  config.getboolean('SUPER','drop_last')
            #self.access_time_update_freq=float(config["SUPER"]["access_time_update_freq"])
            self.random_sampling_enabled=config.getboolean('SUPER','use_random_sampling')
            self.look_ahead_distance =int(config["SUPER"]["look_ahead_distance"])
            self.warm_up_distance=int(config["SUPER"]["warm_up_distance"])
            self.redis_port=int(config["SION"]["port"])
            self.redis_host=(config["SION"]["host"])
            self.lambda_func_name=config["SUPER"]["lambda_func_name"]
            self.use_lambda = config.getboolean('SUPER','use_lambda')

            return config
        else:
            print(f'Unable to read config')
            return

    def _check_environment(self):
        #check S3 bucket
        if not aws.s3bucket_exists(self.bucket_name):
            print(f'The bucket {self.bucket_name} does not exist or you have no access.')
            return
        #load training dataset S3 bucket
        self.train_dataset = Dataset(self.bucket_name,'train',self.micro_batch_size, drop_last=self.drop_last, use_random_sampling=
                                      self.random_sampling_enabled)
        print(f'Successfully loaded dataset from "{self.bucket_name}/train". Total files:{self.train_dataset.length}')

    def _gen_new_group_of_batches(self):
        self.global_batch_group_idx +=1 #TODO:
        new_batches = self.train_dataset.generate_set_of_batches(self.global_batch_group_idx)
        self.batch_groups[self.global_batch_group_idx] = BatchGroup(self.global_batch_group_idx,new_batches,self.lambda_wrapper,self.redis_client, self.use_lambda)

    
    def RegisterNewTrainingJob(self, request, context):
        job_id = request.job_id
        batch_size = request.batch_size 
        
        newjob = MLTrainingJob(job_id,batch_size,self.look_ahead_distance,
                               self.warm_up_distance,self.global_queue,self.use_substitutional_hits)

        if self.drop_last:
            batches_per_epoch = len(self.train_dataset) // batch_size  # type: ignore[arg-type]
        else:
            batches_per_epoch = (len(self.train_dataset) + batch_size - 1) // batch_size  # type: ignore[arg-type]

        if job_id not in self.active_training_jobs:
            self.active_training_jobs[job_id] = newjob
            result = f'Job with Id {job_id} has been successfully regsistered!'
            result = {'message': result, 'dataset':json.dumps(self.train_dataset._blob_classes), 'registered': True,'batches_per_epoch': batches_per_epoch}    
        else:
            result = f'Job with Id {job_id} already exists!'
            result = {'message': result, 'dataset':'[]','registered': False,'batches_per_epoch': batches_per_epoch}
        return pb2.RegisterTrainingJobResponse(**result)
    
    def GetNextBatchForJob(self,request, context):
        training_job:MLTrainingJob = self.active_training_jobs[request.job_id]
        training_job.set_avg_training_speed(request.avg_training_speed)
        #training_job.avg_speed_on_miss = request.prev_batch_dl_delay
        training_job.avg_delay_on_hit = request.avg_delay_on_hit
        training_job.avg_delay_on_miss = request.avg_delay_on_miss
        #training_job.update_data_laoding_delay(request.prev_batch_dl_delay)
        
        if len(training_job.currEpoch_remainingBatches) < 1: #batches exhausted, starting a new epoch
            self._assign_new_batches_to_job_for_epoch(training_job)
            training_job.reset_dl_delay()

        next_batch_id, cache_hit, batch_data, next_batch_incides = training_job.find_next_batch()
        
        result = {'batch_id': str(next_batch_id),'batch_metadata': json.dumps(next_batch_incides), 'isCached': cache_hit,'batch_data': batch_data}
        #logging.info({'batch_id': str(next_batch_id),'batch_metadata': json.dumps(next_batch_incides), 'isCached': cache_hit})
        #logging.info("{} given to job {}. Hit = {}".format(next_batch_id, request.job_id, cache_hit))
        return pb2.GetNextBatchForJobResponse(**result)
    
    def _assign_new_batches_to_job_for_epoch(self,training_job:MLTrainingJob):
        if training_job.currEpoch_batchGroup is not None:
            self.batch_groups[training_job.currEpoch_batchGroup.group_id].processed_by.append(training_job.job_id)

        if  training_job.job_id in self.batch_groups[self.global_batch_group_idx].processed_by:
            self._gen_new_group_of_batches()
        
        training_job.set_batches_for_new_epoch(self.batch_groups[self.global_batch_group_idx])
        training_job.increment_epochs_processed()
    
    def ProcessJobEndedMessage(self, request, context):
        job_id = request.job_id
        job:MLTrainingJob = self.active_training_jobs[job_id]
        start_time, finish_time, duration = job.end_job()
        del self.active_training_jobs[job_id]

        result = f'Job with Id {job_id} finished! start time: {start_time}, finish time: {finish_time}, duration: {duration}'
        result = {'message': result, 'received': True}
        return pb2.MessageResponse(**result)

    def GetServerResponse(self, request, context):
        # get the string from the incoming request
        message = request.message
        result = f'Hello I am up and running received "{message}" message from you'
        result = {'message': result, 'received': True}
        return pb2.MessageResponse(**result)
    

def serve():
    port = "50052"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_CacheManagementServiceServicer_to_server(CacheManagementService(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(filename='super.log', encoding='utf-8', level=logging.INFO, 
                            format='%(asctime)s\t%(levelname)s\t%(message)s')
    serve()