from configparser import ConfigParser
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = dir_path+"/config.ini"
# check if the config file exists
exists = os.path.exists(config_filepath)
config = None
if exists:
    print("--------config.ini file found at ", config_filepath)
    config = ConfigParser()
    config.read(config_filepath)
else:
    print("---------config.ini file not found at ", config_filepath)
    
# Retrieve config details
database_config = config["DATABASE"]
s3_config = config["S3"]
# Never print config data in console when working on real projects
print(database_config["host"])
print(database_config["port"])
print(database_config["username"])
print(database_config["password"])
print(database_config["database_name"])
print(database_config["pool_size"])

print(s3_config["bucket"])
print(s3_config["key"])
print(s3_config["secret"])
print(s3_config["region"])