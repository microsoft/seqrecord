# Meeting notes

## Meeting with azure 02/15/2023

Key takeways:
- avoid multiple reads of the same file, the system is designed for parallel reads of multiple files
    - difference processes, workers (GPU) read different files
- use ram as cache (using their api?), using their prefetch system?
- try to sequeeze reads at boundaries of epoches
- load as much data into ram as possible (800Gb ram)
- cache size v.s. file size, a cache ideally should hold several files
- bandwith: 24Gb/s per storage account


the operating system is not designed for loading the whole file into memory. So a more suitable approach would be two threading (producer-consumer model), reading in a block way.


## todo

add parallel write and gather
