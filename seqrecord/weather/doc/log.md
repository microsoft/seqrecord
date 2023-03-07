# dev logs for weather data

## plan
todo:
- [x] compare different approaches for reading frame pairs, including alternative approaches that have not been implemented

- [ ] tune record file size and write buffer size (find support docs)

- [ ] set up testing framework
- [ ] add pre-put worker for writing



## stats

### write

|resolution| record file size | writer buffer size | preput| speed|
|  ---      | ---             | ----                | --- | --- |
|1.4025    |500mb   | 50 mb | False |  1.6min per npy file|
|1.4025    |1G   | 1G | False |  1.5min per npy file|

to


### read

#### Bytes per second

date: 2/15/2023

setup: single dataset `mpi-esm` from `1.40625deg_equally_np_all_levels`
num_gpus: 2 (V100 with 16Gb memory each), num_workers: 4 (on a 12 core machine).


|env| read method | bytes/s|
|---| --- | ---|
|local vm|origin, np.load(npz) |460913615.40424377 |
|local vm|wseqrecord, np.memmap|3729921494.4032807|
|local vm|wseqrecord, np.memmap with preftech|3549654683.913067|
|local vm|wseqrecord, asyn.read|284919930.11675245|
                            


##### notes:
- memmap is not a fair comparison since it is not a complete cycle of disk->memory transfer
- `async read` is not optimized (nor fair) compared to `npy.load()` since `npy.load()` will load data (features) not going to be used
prefetch buffer length


## training speed

setup:
- dataset 50 1.04625 npy files, about 3% of the whole dataset
- machine: 2 V100(16Gb) gpu, 12 cpu cores
- model size is smaller to train on v100, see https://github.com/AutonomousSystemsResearch/climate_pretraining/blob/shc/test_dataloader/configs/test_dataload_v100.yaml
- dataloader:
    - number of workers: 4 (6 is too big, worker gets killed)
    - batch size 64
    - pin memory false

| datamodule | notes| one training epoch | estimate for one training epoch of the whole dataset | env |
| ---        | ---  | ---                | ---                                                  | --- |
| wseqrecord | memmap| 2min53s      | 98 mins                                              | local vm|
| wseqrecord | memmap| 2min53s      | 98 mins                                              | aml (2v100)|
| wseqrecord | memmap from sharded files| 1min28s      | 98 mins                                              | aml (2v100)|
| wseqrecord | memmap with prefetch==10| 3min10seconds      | 104 mins                           | local vm|
| wseqrecord | asycn read two frames (each feature with separate reads)| 5min10seconds      | 170 mins                           | local vm|
| wseqrecord | asycn read two frames (read all features in one time)| 3min25seconds      | 113 mins                           | local vm|
|native npy reader| none | 1min21s | 39 min|  local vm|
|native npy reader| none | 1min08s | 37 min|  aml (2v100)|