"""transform existing weather dataset to sequence weather format.
"""
from seqrecord.weather.seqrecord import WSeqRecord
import os
from typing import List
import re
import numpy as np
from tqdm import tqdm

# test correctness of saved seqrecord format

# dataset: 5.625deg_equally_np_all_levels,  1.40625deg_equally_np_all_levels(ultimate goal)
DEFAULT_CONFIG = {
    "dataset_mount_dir": "/datadrive/weatherdatastorage2/datasets",  # "/datadrive/weatherdatastorage2/datasets",  # "/mnt/data",
    "wsrecord_dir": "CMIP6/MPI-ESM/wseqrecord/test/1.40625deg_equally_np_all_levels/train/",
    "wdataset_dir": "CMIP6/MPI-ESM/1.40625deg_equally_np_all_levels/train/",
    "wdataset_split": "train",
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


def main(config: dict) -> None:

    wsrecord = WSeqRecord(
        os.path.join(config["dataset_mount_dir"], config["wsrecord_dir"])
    )

    files_dirs = os.scandir(
        os.path.join(config["dataset_mount_dir"], config["wdataset_dir"])
    )
    files = list(filter(lambda direntry: direntry.is_file(), files_dirs))
    files = sort_weatherfiles(files)

    def frame_generator(files):
        i = 0
        for file_path in tqdm(files):
            if i >= 50:
                break
            data = np.load(file_path)
            num_frames = data[data.files[0]].shape[0]
            for rel_frame_idx in range(num_frames):
                frame = {}
                for key in data.files:
                    frame[key] = data[key][rel_frame_idx]
                yield frame
            i += 1

    wsrecord.put_frame(frame_generator(files), 5)
    wsrecord.dump()


if __name__ == "__main__":

    config = DEFAULT_CONFIG
    main(config)
