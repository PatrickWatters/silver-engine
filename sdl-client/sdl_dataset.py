import json
import torch
import functools
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
from PIL import Image
#from misc.time_helper import stopwatch
from torch.utils.data import Dataset
import io
from client import CMSClient
import base64
import time
import gzip

client=CMSClient()

class SDLDataset(Dataset):
    def __init__(self,job_id:int,length:int, transform:Optional[Callable] =None,target_transform:Optional[Callable]=None):
        self.transform = transform
        self.target_transform = target_transform
        self.job_id = job_id
        self.lenth = length

    def __len__(self) -> int:
        return self.lenth
    
    def __getitem__(self, input):
        batch_data = input.data
        batch_id = input.batchid
        cache_hit = True
        end = time.time()
        torch_imgs, torch_lables = self.deserialize_torch_bacth(batch_data)
        #torch_imgs, torch_lables = self.convert_json_batch_to_torch_format(batch_data)
        prep_time = time.time() - end
        return torch_imgs, torch_lables, batch_id, cache_hit, prep_time
    
    def deserialize_torch_bacth(self,batch_data):
        batch_data = base64.b64decode(batch_data)
        decompressed = gzip.decompress(batch_data)
        buffer = io.BytesIO(decompressed)
        decoded_batch = torch.load(buffer)
        batch_imgs = decoded_batch['inputs']
        batch_labels = decoded_batch['labels']
        return batch_imgs,batch_labels


    def convert_json_batch_to_torch_format(self,batch_data):
        samples = json.loads(batch_data)
        imgs = []
        labels  =[]
        
        for img,label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)

            imgs.append(img)
            labels.append(label)

        return torch.stack(imgs), torch.tensor(labels)

'''
def __getitem__(self, input):
        batch_id = input[0]
        batch_indices = input[1]
        isCached = input[2]
        data = input[3]

        batch_data = redis_client.get_data(batch_id)
        if batch_data != None:
            cache_hit = True
            buffer = io.BytesIO(batch_data)
            batch_with_labels_dict = torch.load(buffer)
            return batch_with_labels_dict['inputs'],batch_with_labels_dict['labels'], batch_id,cache_hit
        else:
            return None, None, batch_id, isCached
'''