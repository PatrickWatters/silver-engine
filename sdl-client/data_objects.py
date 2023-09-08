import dataclasses

@dataclasses.dataclass
class BatchMeasurment:
    JobId :int
    BatchId :int
    Epoch: int
    BatchIdx: int
    NumFiles:int
    TotalBatchTime: float
    ImgsPerSec: float
    DataFetchTime: float
    DataPrepTime: float
    TransferToGpuTime: float
    ProcessingTime: float
    Loss: float
    Acc1: float
    Acc5: float
    CacheHit: bool

@dataclasses.dataclass
class EpochMeasurment:
    JobId :int
    Epoch: int
    NumBatches: int
    NumFiles:int
    TotalEpochTime: float
    ImgsPerSec: float
    BatchesPerSec: float

    DataFetchTime: float
    DataPrepTime: float
    TransferToGpuTime: float
    ProcessingTime: float

    AvgLoss: dict
    AvgAcc1: dict
    AvgAcc5: dict

    AvgBatchTime: float
    AvgDataFetchTime: float
    AvgDataPrepTime: float
    AvgTransferToGpuTime: float
    AvgProcessingTime: float
    TotalCacheHits:int
    TotaCacheMisses:int
    CacheHitPercentage: float

@dataclasses.dataclass
class JobMeasurment:
    JobId :int
    Dataset: str
    Model: str
    BacthSize: str
    GPUCount: int

    TotalEpochs: int =0
    TotalBatches: int =0
    TotalFiles:int =0
    JobTime: float =0


    ImgsPerSec:float =0
    BatchesPerSec: float=0
    EpochPerMin: float=0
    TotalBatchFetchTime: float=0
    TotalBatchPrepTime: float=0
    TotalTransferToGpuTime: float=0
    TotalProcessingTime: float=0

    Loss: dict=None
    Acc1: dict=None
    Acc5: dict=None

    AvgBatchTime: float=0
    AvgEpochTime: float=0
    
    AvgBatchFetchTime: float=0
    AvgBatchPrepTime: float=0
    AvgTransferToGpuTime: float=0
    AvgProcessingTime: float=0

    TotalCacheHits:int=0
    TotaCacheMisses:int=0
    CacheHitPercentage: float=0

import dataclasses

@dataclasses.dataclass
class TrialMeasurment:
    TrialId: int
    TotalJobs: int
    TotalEpochs: int
    TotalBatches: int
    TotalFiles:int
    TotalTime: float
    
    ImgsPerSec:float
    BatchesPerSec: float
    EpochPerMiin: float
    JobsPerSec: float


    TotalBatchFetchTime: float
    TotalBatchPrepTime: float
    TotalTransferToGpuTime: float
    TotalProcessingTime: float
    Loss: float
    Acc1: float
    Acc5: float
    
    AvgBatchTime: float
    AvgEpochTime: float
    AvgJobTime: float

    AvgBatchFetchTime: float
    AvgBatchPrepTime: float
    AvgTransferToGpuTime: float
    AvgProcessingTime: float

    TotalCacheHits:int
    TotaCacheMisses:int
    CacheHitPercentage: float