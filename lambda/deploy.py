import logging
import base64
import json
import logging
import boto3
from misc.custom_waiter import CustomWaiter,WaitState
from misc.lambda_basics import LambdaWrapper
from misc.retries import wait
import misc.question as q

logger = logging.getLogger(__name__)

class UpdateFunctionWaiter(CustomWaiter):
    """A custom waiter that waits until a function is successfully updated."""
    def __init__(self, client):
        super().__init__(
            'UpdateSuccess', 'GetFunction',
            'Configuration.LastUpdateStatus',
            {'Successful': WaitState.SUCCESS, 'Failed': WaitState.FAILURE},
            client)

    def wait(self, function_name):
        self._wait(FunctionName=function_name)

class SUPERLambdaMgmt(object):

    def __init__(self, handler_code_file='lambda/handlers/lambda_handler_dataload.py',lambda_name='lambda_dataloader', lambda_client=boto3.client('lambda'),iam_resource=boto3.resource('iam')):
        self.lambda_client = lambda_client
        self.iam_resource =iam_resource
        self.handler_code_file =handler_code_file
        self.lambda_name =lambda_name
        self.wrapper = LambdaWrapper(self.lambda_client, self.iam_resource)
    
    def deploy_function(self):
        print('-'*88)
        print("Checking for IAM role for Lambda...")
        iam_role, should_wait = self.wrapper.create_iam_role_for_lambda(self.lambda_name)
        print("Checking for IAM role for Lambda...")
        if should_wait:
            logger.info("Giving AWS time to create resources...")
            wait(10)  
        print(f"Looking for function {self.lambda_name}...")
        function = self.wrapper.get_function(self.lambda_name)
        if function is None:
            print("Zipping the Python script into a deployment package...")
            deployment_package = self.wrapper.create_deployment_package(self.handler_code_file, f"{self.lambda_name}.py")
            print(f"...and creating the {self.lambda_name} Lambda function.")
            self.wrapper.create_function(
            self.lambda_name, f'{self.lambda_name}.lambda_handler', iam_role, deployment_package)
            print(f"Function {self.lambda_name} created.")
        else:
            print(f"Function {self.lambda_name} already exists.")
    print('-'*88)

    def update_function(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        print('-'*88)
        print('-'*88)
        print("Creating a new deployment package...")
        deployment_package = self.wrapper.create_deployment_package(self.handler_code_file, f"{self.lambda_name}.py")
        print(f"...and updating the {self.lambda_name} Lambda function.")
        update_waiter = UpdateFunctionWaiter(self.lambda_client)
        self.wrapper.update_function_code(self.lambda_name, deployment_package)
        update_waiter.wait(self.lambda_name)
        print(f"This function uses an environment variable to control logging level.")
        print(f"Let's set it to DEBUG to get the most logging.")
        self.wrapper.update_function_configuration(
        self.lambda_name, {'LOG_LEVEL': logging.getLevelName(logging.DEBUG)})
    
    def delete_function(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        print('-'*88)
        print('-'*88)
        self.iam_role, should_wait = self.wrapper.create_iam_role_for_lambda(self.lambda_name)
        if should_wait:
            logger.info("Giving AWS time to create resources...")
            wait(10)
        if q.ask("Ready to delete the function and role? (y/n) ", q.is_yesno):
            self.wrapper.delete_function(self.lambda_name)
            print(f"Deleted function {self.lambda_name}.")
    
    def invoke_function(self,action_params, get_log =False):
        #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        #function = self.wrapper.get_function(self.lambda_name)
        #if function is None:
            #print(f"Function {self.lambda_name} doesn't exists.")
        #else:
            #print(f"Let's invoke {self.lambda_name}")
            #print(f"Invoking {self.lambda_name}...")
            response = self.wrapper.invoke_function(self.lambda_name, action_params,get_log)
            #print(f"Result:" f"{json.load(response['Payload'])}")  
            return json.load(response['Payload'])
        #print('-'*88)

    
if __name__ == '__main__':
    super_lambda = SUPERLambdaMgmt(handler_code_file='/Users/patrickwatters/Projects/super/lambda/handlers/lambda_handler_dataload.py',lambda_name='lambda_dataloader')
    #super_lambda.deploy_function()
    super_lambda.update_function()
    print('')
   