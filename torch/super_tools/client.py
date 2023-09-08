import grpc
import data_feed_pb2 as data_feed_pb2
import data_feed_pb2_grpc as data_feed_pb2_grpc
import time
import csv
import torch
import torchvision.transforms as transforms
import json
from PIL import Image
import io
import base64
import gzip
import os
import multiprocessing
import time

class CMSClient(object):
    """
    Client for gRPC functionality
    """
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50052

        # instantiate a channe
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = data_feed_pb2_grpc.DatasetFeedStub(self.channel)

    def registerJob(self,job_id:int):
        messageRequest = data_feed_pb2.RegisterJobRequest(job_id=job_id)
        response = self.stub.registerjob(messageRequest)
        #response = self.stub.registerjob(messageRequest)
        return response.message, response.successfully_registered,response.batches_per_epoch,response.dataset_length
    
    def getBatches(self,job_id:int):
        messageRequest = data_feed_pb2.GetBatchesRequest(job_id=job_id)
        response = self.stub.getBatches(messageRequest)
        return response


