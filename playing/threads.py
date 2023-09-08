# SuperFastPython.com
# example of stopping daemon thread background task
from time import sleep
from threading import Thread
from threading import Event
 
# background task
def task(event):
    # run until the event is set
    while not event.is_set():
        # run every 2 seconds
        sleep(2)
        # perform task
        print('Background performing task')
    print('Background done')
 
# prepare state for stopping the background
stop_event = Event()
# create and start the background thread
thread = Thread(target=task, args=(stop_event,), daemon=True, name="Background")
thread.start()
# run the main thread for a while
print('Main thread running...')
sleep(10)
print('Main thread stopping')
# request the background thread stop
stop_event.set()
# wait for the background thread to stop
stop_event.clear()
thread = Thread(target=task, args=(stop_event,), daemon=True, name="Background")
thread.start()
thread.join()
print('Main done')