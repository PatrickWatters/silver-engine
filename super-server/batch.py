from threading import Thread, Lock
import time

class Batch():
    def __init__(self,batchId:int, setId:int,indices,labelled_paths):
        self.batchId = batchId
        self.indices = indices
        self.last_pinged = None
        self.processed_by =[]
        self.isCached = False
        self.setId = setId
        self.labelled_paths = labelled_paths
        self.inProgress = False
        self.count = 0
        self.lock = Lock()
        self.batch_type = None

    def setInProgessStatus(self, isInProgress=False):
        with self.lock:
            self.inProgress = isInProgress
    
    def setCachedStatus(self, isCached=False):
        with self.lock:
            self.isCached = isCached
    
    def updateLastPinged(self):
        with self.lock:
            self.last_pinged = time.time()
    
    def updateProcessedBy(self, jobId):
        with self.lock:
            self.processed_by.append(jobId)
    
    def incrementCounter(self):
        with self.lock:
            for _ in range(100000):
                self.count += 1
            #print(self.counter)
    
class BatchSet():
     
     def __init__(self,setId, train_bacthes:dict[int,Batch]=None, val_batches:dict[int,Batch]=None):
        self.setId = setId
        self.batches:dict[int,Batch] =train_bacthes | val_batches
        self.train_batches_ids = train_bacthes.keys()
        self.val_batches_ids = val_batches.keys()
        self.finshedProcessing =[] #the jobs that have finished processing this set of bacthes
        self.isActive = True


if __name__ == "__main__":
    numThreads = 5
    threads = [0] * numThreads
    counter = Batch(1,1,None,None)

    for i in range(0, numThreads):
        threads[i] = Thread(target=counter.incrementCounter)

    for i in range(0, numThreads):
        threads[i].start()

    for i in range(0, numThreads):
        threads[i].join()

    if counter.count != 500000:
        print(" count = {0}".format(counter.count), flush=True)
    else:
        print(" count = 50,000 - Try re-running the program.")
