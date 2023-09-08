import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from PIL import Image
import json
import base64
import boto3
import torch
import torchvision.transforms as transforms
import numpy as np
from boto3.s3.transfer import TransferConfig
import pickle
import sys
import zlib
import gzip
transform=transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            #transforms.GaussianBlur(11)
            ])
GB = 1024 ** 3

s3 = boto3.client('s3')
config = TransferConfig(use_threads=True,max_concurrency=10,multipart_threshold=5*GB) # To consume less downstream bandwidth, decrease the maximum concurrency

def function_test(batch_labelledItems, rtype = 'base64', s3_bucket='sdl-cifar10'):
    s3_bucket=s3_bucket
    if rtype  == 'torch':
        imgs =[]
        labels =[]
        for key, result in fetch_data_from_s3(s3_bucket,batch_labelledItems,load_img_as_tensor):
            imgs.append(result[0])
            labels.append(result[1])

        with io.BytesIO() as f:
            torchimgs = torch.stack(imgs)
            
            print("Orgional Tensor Batch:{} bytes".format(size_of_tensor_in_bytes(torchimgs)))
            torch.save({'inputs': torch.stack(imgs), 'labels': torch.tensor(labels)}, f)
            
            print("BytesIO Batch before compression:{} bytes".format(sys.getsizeof(f.getvalue())))

            compressed = gzip.compress(f.getvalue(),compresslevel=9)
            print("BytesIO Batch after compression {} bytes".format(sys.getsizeof(compressed)))
            
            base_64_encoded_batch = base64.b64encode(compressed).decode('utf-8')
            print("Base64 batch (with compressed batch):{} bytes".format(sys.getsizeof(base_64_encoded_batch)))
            
            base_64_encoded_batch_no = base64.b64encode(f.getvalue()).decode('utf-8')
            print("Base64 batch (without compressed batch):{} bytes".format(sys.getsizeof(base_64_encoded_batch_no)))
        return base_64_encoded_batch
    
    elif rtype  == 'base64':
        samples =[]
        for key, result in fetch_data_from_s3(s3_bucket,batch_labelledItems, load_img_as_base64):
            samples.append(result)
        json_samples = json.dumps(samples)
        
        print("Non Tensor Batch Size:{} bytes".format(sys.getsizeof(json_samples)))

        return json_samples


def tensor_to_string(data):
    buff = io.BytesIO()
    torch.save(data, buff)
    tensor_as_string = base64.b64encode(buff.getvalue()).decode('utf-8')
    decode =  tensor_as_string.decode("utf-8")
    return decode

def fetch_data_from_s3(s3_bucket,batch_labelledItems,func_name):
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_key = {executor.submit(func_name, key,s3_bucket): key for key in batch_labelledItems}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()
            if not exception:
                yield key, future.result()
            else:
                yield key, exception

def size_of_tensor_in_bytes(encoding):
    return (encoding.nelement() * encoding.element_size())

def load_img_as_base64(bacth_item, s3_bucket):
    file_path = bacth_item[0]
    label = bacth_item[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        base_64_encoded_img = base64.b64encode(f.getvalue()).decode('utf-8')
    return base_64_encoded_img, label


def load_img_as_numpy(labelledItem, s3_bucket):
    file_path = labelledItem[0]
    label = labelledItem[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        img = Image.open(io.BytesIO(f.getvalue()))
        pil_img = img.convert('RGB')
    return np.array(img)

def load_img_as_PIL(labelledItem, s3_bucket):
    file_path = labelledItem[0]
    label = labelledItem[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        img = Image.open(io.BytesIO(f.getvalue()))
        pil_img = img.convert('RGB')
    return pil_img, label

def load_img_as_tensor(labelledItem, s3_bucket):
    file_path = labelledItem[0]
    label = labelledItem[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        img = Image.open(io.BytesIO(f.getvalue()))
        pil_img = img.convert('RGB')
        tensor_img = transform(pil_img)
    return tensor_img, label


if __name__ == "__main__":
    #create some dummy data
    #rtype  = 'bytes'
    #rtype  = 'torch'
    #rtype  = 'base64'
    batch_labelledItems = []

    for i in range(0,256):
        batch_labelledItems.append(('test/Frog/leopard_frog_s_001876.png',6))
    
    for rtype in ['torch','base64']:
        
        lambda_response = function_test(batch_labelledItems, rtype)
        if rtype  == 'torch':
            #torch response type as base 64 encoded string
            batch_data = base64.b64decode(lambda_response)
            decompressed = gzip.decompress(batch_data)
            buffer = io.BytesIO(decompressed)
            decoded_batch = torch.load(buffer)
            batch_imgs = decoded_batch['inputs']
            batch_labels = decoded_batch['labels']
            print("Final Tensor Size:{} bytes".format(size_of_tensor_in_bytes(batch_imgs)))

        elif rtype  == 'base64':
            samples = json.loads(lambda_response)
            tensor_imgs = []
            labels  =[]
            for img,label in samples:
                pil_img = Image.open(io.BytesIO(base64.b64decode(img)))
                tensor_img = transform(pil_img)
                tensor_imgs.append(tensor_img)
                labels.append(label)
            batch_imgs = torch.stack(tensor_imgs)
            batch_labels = torch.tensor(labels)
            print("Final Tensor Size:{} bytes".format(size_of_tensor_in_bytes(batch_imgs)))
