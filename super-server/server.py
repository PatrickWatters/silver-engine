# Imports Python builtins.
from concurrent import futures
import multiprocessing as mp
import logging
import sys
from typing import Dict

# Imports grpc realated packages.
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
# Imports local packages.
from args import parse_args
from job import MLTrainingJob

# Logging initialization
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class DatasetFeedService(data_feed_pb2_grpc.DatasetFeedServicer):

    def __init__(self, kill_event,args):
        from coordinator import DataFeedCoordinator
        self.args = args
        self.coordinator = DataFeedCoordinator(self.args)
        self.training_jobs:Dict[str, MLTrainingJob] = {}
        self.kill_event = kill_event

        self.training_jobs[1] = MLTrainingJob(job_id=1, 
                                                   coordinator=self.coordinator,
                                                   args=self.args)
        self.training_jobs[1].start_data_prep_workers()


    def shutdown(self, request, context):
        logger.info("Received shutdown request - Not implemented")
        # from main_grpc_client import shutdown_data_service
        # shutdown_data_service()
        context.set_code(grpc.StatusCode.OK)
        context.set_details('Shutting down')
        return data_feed_pb2.Empty()
        
    def registerjob(self, request, context):
        job_id = request.job_id
        if job_id in self.training_jobs:
            return data_feed_pb2.RegisterJobResponse(message="Job {} not regsistered".format(job_id),
                                                     successfully_registered = False,
                                                     batches_per_epoch=self.coordinator.train_daatset.batches_per_epoch,
                                                     dataset_length = len(self.coordinator.train_daatset))
        else:
            self.training_jobs[job_id] = MLTrainingJob(job_id=job_id, 
                                                   coordinator=self.coordinator,
                                                   args=self.args)
            self.training_jobs[job_id].start_data_prep_workers()
            return data_feed_pb2.RegisterJobResponse(message="Job {} successfully regsistered".format(job_id),
                                                     successfully_registered = True,
                                                     batches_per_epoch=self.coordinator.batches_per_epoch,
                                                     dataset_length = len(self.coordinator))
    
    def getBatch(self, request, context):
        job_id = request.job_id
        split = request.split
        batch_data, batch_id = self.training_jobs[job_id].next_batch(split_type=split)
        return data_feed_pb2.Sample(batchid=batch_id,
                                    data = batch_data,
                                    label = str(batch_id))
def start(kill_event, args):
    server = grpc.server(futures.ThreadPoolExecutor(args.grpc_maxworkers))
    data_feed_pb2_grpc.add_DatasetFeedServicer_to_server(
        DatasetFeedService(kill_event,args), server)
    server.add_insecure_port("[::]:" + args.gprc_port)
    server.start()
    logger.info("gRPC Data Server started, listening on port {}.".format(args.gprc_port))
    return server

def shutdown(grpc_server):
    logger.info('Shutting down...')
    logger.info('Stopping gRPC server...')
    grpc_server.stop(2).wait()
    logger.info('Shutdown done.')
    import os, time
    os.system('kill -9 %d' % os.getpid())

def wait_for_shutdown_signal():
    SHUTDOWN_PORT = 16000
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', SHUTDOWN_PORT))
    s.listen(1)
    #logger.info('Awaiting shutdown signal on port {}'.format(SHUTDOWN_PORT))
    conn, addr = s.accept()
    print('Received shutdown signal from: ', addr)
    try:
        conn.close()
        s.close()
    except Exception as e:
        logger.info(e)
        
def serve(args):
    kill_event = mp.Event() # an mp.Event for graceful shutdown
    grpc_server = start(kill_event, args)
    wait_for_shutdown_signal()
    kill_event.set()
    shutdown(grpc_server)

if __name__ == "__main__":
    #args = parse_args(default_config_file='/home/ubuntu/super/sdl-server/cfgs/cifar10_local.yaml')
    args = parse_args(default_config_file='super-server/cfgs/local_cifar10.yaml')
    print(args)
    serve(args)