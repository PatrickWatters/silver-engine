from queue import Queue
import threading
import time
import random

class ConsumerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        #while True:
            while not q.empty():
                item = q.get()
                print(self.getName() + ' getting ' + str(item)  + ' : ' + str(q.qsize()) + ' items in queue')
                time.sleep(random.random())
            return
    
if __name__ == '__main__':
    consumers = []
    q = Queue()

    for i in range(0, 100):
       q.put(i)

    end = time.time()
    for i in range(16):
        name = 'Consumer-{}'.format(i)
        c = ConsumerThread(name=name)
        c.start()
        consumers.append(c)

    for consumer in consumers:
        consumer.join()

    print('ended',time.time() - end)
      