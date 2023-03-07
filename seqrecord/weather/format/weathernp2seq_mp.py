"""transform existing weather dataset to sequence weather format.

notes:
    - assume the entire dataset is stored in the single storage account.
"""
from seqrecord.weather.seqrecord import WSeqRecord
import os
from typing import List, Tuple, Iterator
import re
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from itertools import islice

# test correctness of saved seqrecord format

# dataset: 5.625deg_equally_np_all_levels,  1.40625deg_equally_np_all_levels(ultimate goal)
dataset_mount_dir = "/datadrive/weatherdatastorage2/datasets"
# "/datadrive/weatherdatastorage2/datasets",  # "/mnt/data",
DEFAULT_CONFIG = {
    "num_processes": 4,
    "wseqrecord_dir": os.path.join(
        dataset_mount_dir,
        "CMIP6/MPI-ESM/wseqrecord/dev/5.625deg_equally_np_all_levels/train/",
    ),
    "wdataset_dir": os.path.join(
        dataset_mount_dir, "CMIP6/MPI-ESM/5.625deg_equally_np_all_levels/train/"
    ),
}


def sort_weatherfiles(files: List[os.DirEntry]) -> List[os.DirEntry]:
    """Return sorted files in dir, make sure files are sorted incrementally in time.
    # todo: check with Jayesh this sorting is correct
    Example file names: 195001010600-195501010000_33.npz from CMIP6/MPI-ESM/1.40625deg_equally_np_all_levels/train/

    Args:
        files (List[os.DirEntry]): each element in list is a file name
    """

    def str2nums(direntry):
        nums = re.split("-|_", direntry.name.split(".")[0])
        nums = tuple(map(int, nums))
        return nums

    return sorted(files, key=str2nums)


def weatherdata2seqrecord(
    rank: int,
    world_size: int,
    config: dict,
    sub_weatherfile_generator: Iterator,
    sub_loads: int,
) -> None:

    sub_wsrecord = WSeqRecord(
        os.path.join(
            config["wseqrecord_dir"], WSeqRecord.subfolder_name(rank, world_size)
        )
    )

    def frame_generator(files: Iterator):
        i = 0
        pbar = (
            tqdm(files, desc="Formatting progress on rank 0:", total=sub_loads)
            if rank == 0
            else files
        )
        for file_path in pbar:
            data = np.load(file_path)
            num_frames = data[data.files[0]].shape[0]
            for rel_frame_idx in range(num_frames):
                frame = {}
                for key in data.files:
                    frame[key] = data[key][rel_frame_idx]
                yield frame
            i += 1

    sub_wsrecord.put_frame(frame_generator(sub_weatherfile_generator), 5)
    sub_wsrecord.dump(rank)


def distribute_loads(works: int, num_processes: int) -> List[Tuple[int, int]]:
    """Given the overall works and number of processes, allocate evenly the loads that each process should take.

    Args:
        works (int): amount of over all work
        num_processes (int): number of processes available

    Returns:
        List[Tuple[int, int]]: indices of work each process is responsible for
    """
    assert (
        works >= num_processes
    ), "The amount of works is less than number of processes."
    ans = []
    start = 0
    loads_per_process = round(works / num_processes)
    for i in range(num_processes):
        end = start + loads_per_process if i < num_processes - 1 else works
        ans.append((start, end))
        start = end
    return ans


if __name__ == "__main__":

    config = DEFAULT_CONFIG

    # gather existing weather dataset files
    files_dirs = os.scandir(os.path.join(config["wdataset_dir"]))
    files = list(filter(lambda direntry: direntry.is_file(), files_dirs))
    # make sure files are ordered by time incrementally
    # different datasets require different sortting methods
    files = sort_weatherfiles(files)[:4]
    weatherfile_generator = iter(files)
    dividens = distribute_loads(len(files), config["num_processes"])
    processes = []
    for i in range(config["num_processes"]):
        sub_weatherfile_generator = islice(
            weatherfile_generator, dividens[i][0], dividens[i][1]
        )
        p = Process(
            target=weatherdata2seqrecord,
            args=(
                i,
                config["num_processes"],
                config,
                sub_weatherfile_generator,
                dividens[i][1] - dividens[i][0],
            ),
        )
        processes.append(p)
        p.start()
    # all process complete successfully
    for i in range(config["num_processes"]):
        processes[i].join()

    # combine sub-seqrecord
    record = WSeqRecord.gather_subseqrecords(
        config["wseqrecord_dir"], world_size=config["num_processes"]
    )
