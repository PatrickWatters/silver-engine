import grpc
import data_feed_pb2
import data_feed_pb2_grpc
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

transform=transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.ColorJitter(11)
            ])

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
        return response
    
    def getBatches(self,job_id:int):
        messageRequest = data_feed_pb2.GetBatchesRequest(job_id=job_id)
        response = self.stub.getBatches(messageRequest)
        return response
    
    def getBatchStream(self,job_id:int):
        couneter =0
        messageRequest = data_feed_pb2.GetBatchesRequest(job_id=job_id)
        samples = self.stub.getBatches(messageRequest)
        end = time.time()
        for s in samples:
            couneter+=1
            print(couneter)
            if couneter ==50:
                print('total time: {}'.format(time.time()-end))
                break

def convert_json_batch_to_torch_format(batch_data):
        samples = json.loads(batch_data)
        imgs = []
        labels  =[]
        for img,label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            tensor_img = transform(img)
            imgs.append(tensor_img)
            labels.append(label)
        timgs = torch.stack(imgs)
        tlabels = torch.tensor(labels)
        return size_of_tensor_in_bytes(timgs)

def size_of_tensor_in_bytes(encoding):
    return (encoding.nelement() * encoding.element_size())

def run_local_img_transform_test(numBatches = 5):
    stats =[]
    client = CMSClient()
    job_id= os.getpid()
    response = client.registerJob(job_id)
    count =0
    tsize =0
    totaltansformation_time = 0
    totalfetch_time = 0

    end = time.time()
    for i in range(0,numBatches):
       
       data_fetcher_timer = time.time()
       response = client.getBatches(job_id)
       data_fetch_time = time.time()-data_fetcher_timer
       
       data = response.data
       
       data_transform_timer = time.time()
       bsize = convert_json_batch_to_torch_format(data)
       data_transform_time = time.time()-data_transform_timer
       totaltansformation_time +=data_transform_time
       totalfetch_time +=data_fetch_time
       tsize+=bsize
       
       stats.append((str(i),bsize,data_fetch_time,data_transform_time))
       count+=1 
       print(count,bsize,data_fetch_time,data_transform_time) 

    stats.append(('total',tsize,totalfetch_time,totaltansformation_time))
    print('total time: {}, datafetch:{}, trasnform {}, toal size: {}'.
          format(time.time()-end,totalfetch_time,totaltansformation_time,tsize))
    
    with open('withcache/local_img_transform_5workers.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for value in stats:
            writer.writerow([value[0], value[1], value[2]])

def deserialize_torch_bacth(batch_data):
    batch_data = base64.b64decode(batch_data)
    decompressed = gzip.decompress(batch_data)
    buffer = io.BytesIO(decompressed)
    decoded_batch = torch.load(buffer)
    batch_imgs = decoded_batch['inputs']
    batch_labels = decoded_batch['labels']
    return size_of_tensor_in_bytes(batch_imgs)

def deserialize_torch_bacth(batch_data):
    batch_data = base64.b64decode(batch_data)
    decompressed = gzip.decompress(batch_data)
    buffer = io.BytesIO(decompressed)
    decoded_batch = torch.load(buffer)
    batch_imgs = decoded_batch['inputs']
    batch_labels = decoded_batch['labels']
    return size_of_tensor_in_bytes(batch_imgs)

def run_lambda_transform_test(numBatches = 5 ):
    stats =[]
    client = CMSClient()
    job_id=os.getpid() 
    response = client.registerJob(job_id)
    count =0
    tsize =0
    totaltansformation_time = 0
    totalfetch_time = 0

    end = time.time()
    for i in range(0,numBatches):
       
       data_fetcher_timer = time.time()
       response = client.getBatches(job_id)
       data_fetch_time = time.time()-data_fetcher_timer
       
       data = response.data

       data_transform_timer = time.time()
       bsize = deserialize_torch_bacth(data)
       data_transform_time = time.time()-data_transform_timer
       totaltansformation_time +=data_transform_time
       totalfetch_time +=data_fetch_time
       tsize+=bsize
       
       stats.append((str(i),bsize,data_fetch_time,data_transform_time))
       count+=1 
       print(count,bsize,data_fetch_time,data_transform_time) 

    stats.append(('total',tsize,totalfetch_time,totaltansformation_time))
    print('total time: {}, datafetch:{}, trasnform {}, toal size: {}'.
          format(time.time()-end,totalfetch_time,totaltansformation_time,tsize))
    
    with open('withcache/lambda_img_transform_30workers.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for value in stats:
            writer.writerow([value[0], value[1], value[2],value[3]])


#functions for testing the implementation
if __name__ == '__main__':
    run_local_img_transform_test(500)
    run_lambda_transform_test(500)





