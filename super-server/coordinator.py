from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
import json
from batch import BatchSet, Batch
from torch.utils.data.sampler import RandomSampler,BatchSampler, SequentialSampler
import time
import logging
import sys
import functools
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class S3Url(object):
  
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()



class DataFeedCoordinator():
    def __init__(self,args):
        self.global_epoch_counter = 0
        if 'train_data_dir' in vars(args):
            self.train_dataset = Dataset(args=args,prefix='train',data_dir=args.train_data_dir)
        if 'val_data_dir' in vars(args):
            self.validation_dataset = Dataset(args=args,prefix='val',data_dir=args.val_data_dir)
        self.batchSets:Dict[int, BatchSet] = {}

    def get_batch_type(self,bacthSetId, batchId):
        if batchId in self.batchSets[bacthSetId].train_batches_ids:
            return 'train'
        elif batchId in self.batchSets[bacthSetId].val_batches_ids:
            return 'val'
        
    def batch_is_cached(self, bacthSetId, batchId):
        #check if batch is cached or is in the process of being cached
        return self.batchSets[bacthSetId].batches[batchId].isCached
    
    def batch_is_inProgress(self, bacthSetId, batchId):
        #check if batch is cached or is in the process of being cached
        return self.batchSets[bacthSetId].batches[batchId].isCached
    
    def set_batch_inProgress(self, bacthSetId, batchId, status):
        self.batchSets[bacthSetId].batches[batchId].setInProgessStatus(isInProgress=status)
    
    def set_batch_isCached(self, bacthSetId, batchId, status):
        self.batchSets[bacthSetId].batches[batchId].setCachedStatus(isCached=status)
    
    def update_batch_last_access_time(self, bacthSetId, batchId):
        self.batchSets[bacthSetId].batches[batchId].updateLastPinged()
    
    def get_batch_lablled_paths(self, bacthSetId, batchId):
        return self.batchSets[bacthSetId].batches[batchId].labelled_paths
    
    def new_epoch_batches_for_job(self, jobId, jobBatchSetId):
        if jobBatchSetId is not None:  
            self.batchSets[jobBatchSetId].finshedProcessing.append(jobId)

        if self.global_epoch_counter == 0 or jobId in self.batchSets[self.global_epoch_counter].finshedProcessing:
            #need to generate new set of global batches
            self.global_epoch_counter +=1
            train_batches = {}
            val_batches = {}
            if hasattr(self, 'train_dataset'):
                train_batches = self.train_dataset.gen_batches(
                    self.global_epoch_counter)
            if hasattr(self, 'validation_dataset'):
                val_batches = self.validation_dataset.gen_batches(
                    self.global_epoch_counter)
            
            newBatchSet = BatchSet(self.global_epoch_counter,
                                   train_bacthes=train_batches,
                                   val_batches=val_batches)
            self.batchSets[newBatchSet.setId] = newBatchSet
            return train_batches.keys(), val_batches.keys(),newBatchSet.setId
        else:
         batch_set:BatchSet = self.batchSets[self.global_epoch_counter]
         return batch_set.train_batches_ids,batch_set.val_batches_ids,batch_set.setId


class Dataset():
    def __init__(self,args, prefix, data_dir):
        self.batch_size = args.batch_size
        self.drop_last =args.dataset_droplast
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        self.prefix = prefix
        if 'sampler_seed' in vars(args):
            self.sampler_seed = args.sampler_seed
        else:
            self.sampler_seed =None
        if args.source_system == 's3':
            self._blob_classes = self._classify_blobs_s3(S3Url(data_dir))
        else:
            self._blob_classes = self._classify_blobs_local(data_dir)

        self.batches_per_epoch = (len(self) + self.batch_size -1)// self.batch_size
    
    def gen_batches(self,setid):
        from torch import Generator

        batches={}
        if self.sampler_seed is not None:
            generator = Generator()
            generator.manual_seed(self.sampler_seed)
            base_sampler = RandomSampler(self,generator=generator)
        else:
            base_sampler = RandomSampler(self)
        
        batch_sampler = BatchSampler(base_sampler, batch_size=self.batch_size, drop_last=False)
        
        for i,batchIndiceis in enumerate(batch_sampler):
            batchid = abs(hash(frozenset(batchIndiceis)))
            labelled_paths =[]
            for i in batchIndiceis:
                labelled_paths.append(self._classed_items[i])
                newBatch = Batch(batchId=id,setId=setid,indices=batchIndiceis, labelled_paths=labelled_paths)
            batches[batchid] = newBatch
        return batches

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self._blob_classes.values())
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._blob_classes)
            for blob in self._blob_classes[blob_class]
        ]
    
    def _remove_prefix(self,s: str, prefix: str) -> str:
        if not s.startswith(prefix):
            return s
        return s[len(prefix) :]

    def _classify_blobs_local(self,data_dir) -> Dict[str, List[str]]:
        import os

        logger.info("Reading index files (all the file paths in the data_source)")
        blob_classes: Dict[str, List[str]] = {}
        
        index_file = Path(data_dir + 'index.json')
        
        if(index_file.exists()):
            f = open(index_file.absolute())
            blob_classes = json.load(f)
        else:
            logger.info("No index file found for {}, creating it..".format(data_dir))
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    if not self.is_image_file(filename):
                        continue
                    blob_class = os.path.basename(dirpath.removesuffix('/'))
                    blobs_with_class = blob_classes.get(blob_class, [])
                    blobs_with_class.append(os.path.join(dirpath,filename))
                    blob_classes[blob_class] = blobs_with_class

            json_object = json.dumps(blob_classes, indent=4)
            with open(data_dir + 'index.json', "w") as outfile:
                outfile.write(json_object)

        totalfiles = sum(len(class_items) for class_items in blob_classes.values())
        logger.info("Finished loading {} index. Total files:{}, Total classes:{}".format(self.prefix,totalfiles,len(blob_classes)))
        return blob_classes

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def _classify_blobs_s3(self,s3url:S3Url) -> Dict[str, List[str]]:
        import boto3

        s3_client = boto3.client('s3')
        s3Resource = boto3.resource("s3")
        logger.info("Reading index file (all the file paths in the data_source) for {}.".format(s3url.url))

        #check if 'prefix' folder exists
        resp = s3_client.list_objects(Bucket=s3url.bucket, Prefix=s3url.key, Delimiter='/',MaxKeys=1)
        if not 'NextMarker' in resp:
            logger.info("{} dir not found. Skipping {} task".format(s3url.url, self.prefix))
            return None
        blob_classes: Dict[str, List[str]] = {}
        #check if index file in the root of the folder to avoid having to loop through the entire bucket
        content_object = s3Resource.Object(s3url.bucket, s3url.key + 'index.json')
        try:
            file_content = content_object.get()['Body'].read().decode('utf-8')
            blob_classes = json.loads(file_content) 
        except:
            logger.info("No index file found for {}, creating it..".format(s3url.url))
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3url.bucket, Prefix=s3url.key)        
            for page in pages:
                for blob in page['Contents']:
                    blob_path = blob.get('Key')
                    #check if the object is a folder which we want to ignore
                    if blob_path[-1] == "/":
                        continue
                    stripped_path = self._remove_prefix(blob_path, s3url.key).lstrip("/")
                    #Indicates that it did not match the starting prefix
                    if stripped_path == blob_path:
                        continue
                    if not self.is_image_file(blob_path):
                        continue
                    blob_class = stripped_path.split("/")[0]
                    blobs_with_class = blob_classes.get(blob_class, [])
                    blobs_with_class.append(blob_path)
                    blob_classes[blob_class] = blobs_with_class
                
            s3object = s3Resource.Object(s3url.bucket, s3url.key +'index.json')
            s3object.put(Body=(bytes(json.dumps(blob_classes, indent=4).encode('UTF-8'))))
        
        totalfiles = sum(len(class_items) for class_items in blob_classes.values())
        logger.info("Finished loading {} index. Total files:{}, Total classes:{}".format(self.prefix,totalfiles,len(blob_classes)))
        return blob_classes
    




