import os
import logging
import json
import platform
from data_objects import JobMeasurment, BatchMeasurment, EpochMeasurment, TrialMeasurment
import csv
import io
import pandas as pd
import sys

#formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
#custom_formatter = logging.Formatter('%(asctime)s\t%(message)s')

def setup_logger(name, log_file, level=logging.INFO, 
                 formatter =logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s'),stdout = False):
    """To setup as many loggers as you want"""
    if stdout:
        handler = logging.StreamHandler(stream=sys.stdout)
    else:
        handler = logging.FileHandler(log_file, delay=True)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class TrainingProfiler():
    def __init__(self,args,trailid:int, jobid:int, cuda_device_count =0,):
        self.jobid = jobid
        self.trialid = trailid
        self.output_base_folder = "reports/trail-{}".format(trailid)
        self.trial_file = "{}/summary.tsv".format(self.output_base_folder,trailid)
        self.config_file = "{}/{}-config.json".format(self.output_base_folder,jobid)
        self.batch_stats:list[BatchMeasurment] = []
        self.epoch_stats:list[EpochMeasurment] = []
        self.job_stats:list[JobMeasurment] = []  
        self.dump_job_config(args)
        self.cuda_device_count = cuda_device_count 
        self.job_measurment = JobMeasurment(JobId=jobid, Dataset='cifar10',Model=args.arch, BacthSize= args.batch_size, GPUCount=cuda_device_count)

    def flush_to_execel(self):
        output_file = "{}/job-{}.xlsx".format(self.output_base_folder,self.jobid)
        if len(self.batch_stats) > 0:
            batches_df = pd.DataFrame([t.__dict__ for t in self.batch_stats])
            self.send_to_execl(output_file,batches_df,'Batches')
            self.batch_stats.clear()
        if len(self.epoch_stats) > 0:
            epochs_df = pd.DataFrame([t.__dict__ for t in self.epoch_stats])
            self.send_to_execl(output_file,epochs_df,'Epochs')
            self.epoch_stats.clear()
        if len(self.job_stats) > 0:
            jobs_df = pd.DataFrame([t.__dict__ for t in self.job_stats])
            self.send_to_execl(output_file,jobs_df,'Job')
            self.job_stats.clear()

    def send_to_execl(self, fpath, df:pd.DataFrame, sheet_name):
        if not os.path.exists(fpath):
            with pd.ExcelWriter(fpath, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else: 
            if sheet_name  in pd.ExcelFile(fpath).sheet_names:
                #append to sheeet
                with pd.ExcelWriter(fpath, engine="openpyxl", mode="a",if_sheet_exists="overlay") as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=writer.sheets[sheet_name].max_row, header=False)
            else:
                  with pd.ExcelWriter(fpath, engine="openpyxl", mode="a") as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
         


    def record_batch_stats(self,batch_measurement:BatchMeasurment):
        self.batch_stats.append(batch_measurement)

    
    def record_epoch_stats(self,epoch_measurement:EpochMeasurment):
        self.epoch_stats.append(epoch_measurement)
        self.job_measurment.JobTime += epoch_measurement.TotalEpochTime
        self.job_measurment.JobId = self.jobid
        self.job_measurment.TotalEpochs = epoch_measurement.Epoch
        self.job_measurment.TotalBatches += epoch_measurement.NumBatches
        self.job_measurment.TotalFiles += epoch_measurement.NumFiles
        self.job_measurment.TotalBatchFetchTime += epoch_measurement.DataFetchTime
        self.job_measurment.TotalBatchPrepTime += epoch_measurement.DataPrepTime
        self.job_measurment.TotalTransferToGpuTime += epoch_measurement.TransferToGpuTime
        self.job_measurment.TotalProcessingTime += epoch_measurement.ProcessingTime
        self.job_measurment.TotalCacheHits += epoch_measurement.TotalCacheHits
        self.job_measurment.TotaCacheMisses += epoch_measurement.TotaCacheMisses
        self.job_measurment.Loss = epoch_measurement.AvgLoss
        self.job_measurment.Acc1 = epoch_measurement.AvgAcc1
        self.job_measurment.Acc5 = epoch_measurement.AvgAcc5

    def gen_final_exel_report(self):
        job:JobMeasurment = self.job_measurment 
        job.ImgsPerSec = job.TotalFiles/job.JobTime
        job.BatchesPerSec = job.TotalBatches/job.JobTime
        job.EpochPerMin = (job.TotalEpochs/(job.JobTime/60))
        job.AvgBatchTime = job.JobTime/job.TotalBatches
        job.AvgEpochTime = job.JobTime/job.TotalEpochs
        job.AvgBatchFetchTime = job.TotalBatchFetchTime/job.TotalBatches
        job.AvgBatchPrepTime = job.TotalBatchPrepTime/job.TotalBatches
        job.AvgTransferToGpuTime = job.TotalTransferToGpuTime/job.TotalBatches
        job.AvgProcessingTime = job.TotalProcessingTime/job.TotalBatches
        job.CacheHitPercentage = job.TotalCacheHits/(job.TotalCacheHits+job.TotaCacheMisses)
        self.job_stats.append(job)
        self.flush_to_execel()
        
    def start_gpu_profiler(self):
        if self.cuda_device_count > 0:
            # gpu_util logger
            gpu_util_log = logging.FileHandler(os.path.join(self.output_base_folder, f"gpuutil-{os.getpid()}.log"))
            gpu_util_log.setFormatter(logging.Formatter(""))
            logging.getLogger("gpuutil").addHandler(gpu_util_log)
            logging.getLogger("gpuutil").setLevel(logging.DEBUG)
            logging.getLogger("gpuutil").propagate = False
            #self.gpu_logger = GPUSidecarLogger(refresh_rate=0.5, max_runs=-1)
            print('starting gpu stats logger')
            self.gpu_logger.start()

    def dump_job_config(self,args):
         if not os.path.exists(self.output_base_folder):
             os.makedirs(self.output_base_folder)
         with open(self.config_file, 'w+') as f:
        #with (self.config_file).open("w") as f:
            metadata = vars(args).copy()
            #metadata["output_base_folder"] = metadata["output_base_folder"].name
            metadata.update(platform.uname()._asdict())
            json.dump(metadata, f)
        
    def stop_gpu_profiler(self):
            self.gpu_logger.stop()

    
    def gen_trial_report(self):
        trail_measurment = TrialMeasurment(self.trialid,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        for path in os.listdir(self.output_base_folder):
         if 'job-stats.csv' in path:
             df = pd.read_csv(path)
             jobdata =  pd.read_csv(path).df.to_dict()
             trail_measurment.TotalJobs +=1
             trail_measurment.TotalEpochs += jobdata['TotalEpochs']
             trail_measurment.TotalBatches += jobdata['TotalBatches']
             trail_measurment.TotalFiles += jobdata['TotalFiles']
             trail_measurment.TotalBatchFetchTime += jobdata['TotalBatchFetchTime']
             trail_measurment.TotalBatchPrepTime += jobdata['TotalBatchPrepTime']
             trail_measurment.TotalTransferToGpuTime += jobdata['TotalTransferToGpuTime']
             trail_measurment.TotalProcessingTime += jobdata['TotalProcessingTime']
             trail_measurment.Loss[jobdata['JobId']] = jobdata['Loss']
             trail_measurment.Acc1[jobdata['JobId']] = jobdata['Acc1']
             trail_measurment.Acc5[jobdata['JobId']] = jobdata['Acc5']
             trail_measurment.TotalTime = jobdata['JobTime']
             trail_measurment.TotalCacheHits += jobdata['TotalCacheHits']
             trail_measurment.TotaCacheMisses += jobdata['TotaCacheMisses']
   
        trail_measurment.CacheHitPercentage=trail_measurment.TotalCacheHits/(trail_measurment.TotalCacheHits+trail_measurment.TotaCacheMisses)
        trail_measurment.ImgsPerSec = trail_measurment.TotalFiles/trail_measurment.TotalTime
        trail_measurment.BatchesPerSec = trail_measurment.TotalBatches/trail_measurment.TotalTime
        trail_measurment.EpochPerMiin = (trail_measurment.TotalEpochs/(trail_measurment.TotalTime/60))
        trail_measurment.JobsPerSec = trail_measurment.TotalJobs/trail_measurment.TotalTime
        
        trail_measurment.AvgBatchTime = trail_measurment.TotalTime/trail_measurment.TotalBatches
        trail_measurment.AvgEpochTime = trail_measurment.TotalTime/trail_measurment.TotalEpochs
        trail_measurment.AvgJobTime = trail_measurment.TotalTime/trail_measurment.TotalJobs
        trail_measurment.AvgBatchFetchTime = trail_measurment.TotalBatchFetchTime/trail_measurment.TotalBatches
        trail_measurment.AvgBatchPrepTime = trail_measurment.TotalBatchPrepTime/trail_measurment.TotalBatches
        trail_measurment.AvgTransferToGpuTime = trail_measurment.TotalTransferToGpuTime/trail_measurment.TotalBatches
        trail_measurment.AvgProcessingTime = trail_measurment.TotalProcessingTime/trail_measurment.TotalBatches

        file_path = self.trial_file

        if os.path.exists(file_path):
            os.remove(file_path)
            
        header = list(trail_measurment.__dict__.keys())
        self.write_to_csv(file_path,header)
        row = list(trail_measurment.__dict__.values())
        self.write_to_csv(file_path,row)

    def write_to_csv(self, filepath, row):
        #create file if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            with open(filepath, 'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
