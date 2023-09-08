import random
import time
import torch
import base64
import io
import json
import torchvision.transforms as transforms
from PIL import Image
import logging
import csv
import threading
from queue import Queue
import copy
import botocore

client_config = botocore.config.Config(
    max_pool_connections=64,
)
total_size = 0

def size_of_str(input_string, metric ='bytes'):
    res = len(input_string.encode('utf-8'))
    return convert_bytes(res, metric)

def convert_bytes(byte_value, metric):
    if metric == 'kb':
         return byte_value/1024
    elif metric == 'mb':
         return byte_value/1048576
    elif metric == 'gb':
         return byte_value/1073741824
    else:
         return byte_value

def size_of_tensor_in_bytes(encoding, metric ='bytes'):
    size_in_bytes = encoding.nelement() * encoding.element_size()
    return convert_bytes(size_in_bytes, metric)
 

def process_response(batch_data:str):
    batch_data = base64.b64decode(batch_data)
    buffer = io.BytesIO(batch_data)
    decoded_batch = torch.load(buffer)
    batch_imgs = decoded_batch['inputs']
    batch_labels = decoded_batch['labels']
    
    #samples = json.loads(batch_data)
    #tensor_imgs = []
    #labels  =[]
    #convert_tensor = transforms.ToTensor()
    
    #for img,label in samples:
    #        pil_img = Image.open(io.BytesIO(base64.b64decode(img)))
    #        tensor_imgs.append(convert_tensor(pil_img))
    #        labels.append(label)

    #batch_imgs = torch.stack(batch_imgs)
    #batch_labels = torch.tensor(batch_labels)

    print(size_of_tensor_in_bytes(batch_imgs,'mb'))
    print(size_of_tensor_in_bytes(batch_labels,'mb'))



def gen_dummy_batch(batch_size=256):
    batch_metadata = []
    for i in range(0,batch_size):
        batch_metadata.append(('test/Frog/leopard_frog_s_001876.png',6))
    return batch_metadata

def simple_invokation(action_params,batch_size):
    mgr = SUPERLambdaMgmt(handler_code_file='/Users/patrickwatters/Projects/super/lambda/handlers/lambda_handler_dataload.py',lambda_name='lambda_dataloader')
    print(action_params['batch_id'])
    batch_metadata = gen_dummy_batch(batch_size)
    action_params['batch_metadata'] = batch_metadata

    #mgr.update_function()
    end = time.time()
    #call function
    response = mgr.invoke_function(action_params, False)
    if 'errorMessage' in response:
        print(response['errorMessage'])
        logging.info(response['errorMessage'])
    elif response['isCached'] == False:
        print('Batch Size (bytes):', size_of_str(response['batch_data']))
        print('Response Size (MB):', size_of_str(response['batch_data'],'mb'))
    else:
        if action_params['return_batch_data']:
            print('Batch Size (bytes):', size_of_str(response['batch_data']))
            print('Response Size (MB):', size_of_str(response['batch_data'],'mb'))
            process_response(response['batch_data'])

    print(time.time() - end)

def populate_cache_until_full(action_params,batch_size,outputfile, stop_after = None):
    mgr = SUPERLambdaMgmt(handler_code_file='/Users/patrickwatters/Projects/super/lambda/handlers/lambda_handler_torch_dataload.py',lambda_name='lambda_dataloader_pytorch')
    batch_metadata = gen_dummy_batch(batch_size)
    action_params['batch_metadata'] = batch_metadata
    tend = time.time()
    counter =0
    total_size = 0
    result = []
    while (True):
        if stop_after is not None:
            if counter == stop_after:
                break
        action_params['batch_id'] = counter+1
        single_batch_time = time.time()
        response = mgr.invoke_function(action_params, False)
        if 'errorMessage' in response:
            print(response['errorMessage'])
            logging.info(response['errorMessage'])
            break
        if response['isCached'] == False:
            print('cache error:' + response['cache_error_message'])
            break
        else:
            requesttime =time.time()-single_batch_time
            if action_params['return_batch_data']:
                total_size+=size_of_str(response['batch_data'],'mb')
            else:
                #total_size+= 0.794189453125
                total_size+= 4.00321197509765
            counter +=1
            #total_size +=0.03134
            result.append((action_params['batch_id'],total_size,requesttime,time.time()-tend))
            logging.info((counter,total_size,requesttime,time.time()-tend))

    with open(outputfile, 'w') as f:
        filewriter = csv.writer(f, delimiter='\t',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['BatchId','Total Cached (Mb)','Batch Load Time (s)', 'Elapsed Time (s)'])
        for val in result:
                filewriter.writerow([val[0],val[1],val[2],val[3]])


class ConsumerThread(threading.Thread):
    def __init__(self, group=None, target=None,tend=None, name=None, args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        self.mgr = SUPERLambdaMgmt(handler_code_file='/Users/patrickwatters/Projects/super/lambda/handlers/lambda_handler_torch_dataload.py',lambda_name='lambda_dataloader_pytorch')
        self.total_size =0
        self.tend = tend
        return

    def run(self):
        #while True:
            global total_size
            while not q.empty():
                action_params:dict = q.get()
                batch_id = action_params['batch_id']
                single_batch_time = time.time()
                response = self.mgr.invoke_function(action_params, False)
                requesttime =time.time()-single_batch_time
                total_size +=size_of_str(response['batch_data'],'mb')
                output = (action_params['batch_id'],total_size,requesttime,time.time()-self.tend)
                outq.put(output)
                logging.info((batch_id,total_size,requesttime,time.time()-self.tend))
                #time.sleep(random.random())
            return


if __name__ == '__main__':
    total_size = 0
    consumers = []
    outq = Queue()
    q = Queue()
    
    logging.basicConfig(format='%(asctime)s - %(message)s',filename='cache_load.log', encoding='utf-8', level=logging.INFO)
    batch_metadata = gen_dummy_batch(256)
    action_params = {}
    action_params['batch_metadata'] = batch_metadata
    action_params['batch_id'] = random.randint(0,10000000000)
    action_params['cache_bacth'] = True
    action_params['return_batch_data'] = True
    action_params['bucket'] = 'sdl-cifar10'    
    action_params['redis_host'] = '54.201.114.145'    
    action_params['redis_port'] = 6378
    simple_invokation(action_params,256)
    #populate_cache_until_full(action_params,256,'sion_32_1024mb_laptop_with_no_data_transfer.csv',150)
    
    '''
    
    batch_metadata = gen_dummy_batch(256)
    action_params['batch_metadata'] = batch_metadata
    
    for i in range(300, 600):
        dc = copy.deepcopy(action_params)
        dc['batch_id'] = i+1
        q.put(dc)

    tend = time.time()
    for i in range(64):
        name = 'Consumer-{}'.format(i)
        c = ConsumerThread(tend=tend, name=name)
        c.start()
        consumers.append(c)

    for consumer in consumers:
        consumer.join()

    print('ended',time.time() - tend)

    with open('sion_32_2048mb_ec2_with_data_transfer_multi_threaded64.csv', 'w') as f:
        filewriter = csv.writer(f, delimiter='\t',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['BatchId','Total Cached (Mb)','Batch Load Time (s)', 'Elapsed Time (s)'])
        while not outq.empty():
            val = outq.get()
            filewriter.writerow([val[0],val[1],val[2],val[3]])
    '''