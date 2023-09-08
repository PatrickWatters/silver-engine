from logging import error
import redis
import logging

class RedisClient:

    def __init__(self, redis_host, redis_port ):
        self.exire_time = 0
        self.conn = redis.StrictRedis(host=redis_host, port=redis_port)
        self.isLocal = redis_host == '127.0.0.1'
   

    def set_data(self, key, value):
        try:
            self.conn.set(key, value)
            return True
        except Exception as e:
                logging.error(str(e))
                print(str(e))
                return False
        
    def get_data(self, key):
        try:
            return self.conn.get(key)
    
        except Exception as e:
                logging.error(str(e))
                print(str(e))
                return False