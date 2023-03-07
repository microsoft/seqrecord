"""A package for encoding and decoding weather dataset."""

import asyncio
import copy
import io
import os
import pickle
import random
import threading
import time
from collections import deque
from functools import partial
from time import perf_counter
from typing import (
    Any,
    BinaryIO,
    Deque,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Iterator,
)

import aiofiles
import numpy as np
import yaml
import subprocess
import shutil

from seqrecord.utils import (
    CONSUMER_SLEEP_INTERVAL,
    PRODUCER_SLEEP_INTERVAL,
    FileManager,
    LRUCache,
    TimeTracker,
    WriterBuffer,
)

MAX_RECORDFILE_SIZE = 2e9  # 1e9  # 1e8, 100 mb, maximum size of a single record file
FILE_CACHE_SIZE = 10  # 10, maximum number of record files to keep in local disk
WSR = TypeVar("WSR", bound="WSeqRecord")

# todo: add property stuff for attributes not to be changed by user
# todo: MISSING, some transform work, subsample etc. Need to make sure data is same as produced by existing dataset


class _PrefetchData:
    def __init__(self, source_data_generator, buffer_size: int):
        self.run_prefetcher = True
        # python deque is thread safe for appends and pops from opposite sides.
        # ref: https://stackoverflow.com/questions/8554153/is-this-deque-thread-safe-in-python
        self.prefetch_buffer: Deque = deque()
        self.buffer_size: int = buffer_size
        self.source_data_generator = source_data_generator


class WSeqRecord:
    """A serialization protocal that stores a single continuous long sequence of weather data into record files, while provides metadata of each frame to enable efficient random access of frames."""

    def __init__(self, recorddir: str, local_cache_dir: Optional[str] = None) -> None:
        """_summary_

        Args:
            recorddir (str): directory record files is placed
        """
        # folder that stores data in a separate directory (subfolder)
        self.recorddir: str = recorddir
        os.makedirs(self.recorddir, exist_ok=True)
        # in case we use realtive path '~'
        self.local_cache_dir = local_cache_dir
        if local_cache_dir is not None:
            self.local_cache_dir = os.path.abspath(os.path.expanduser(local_cache_dir))
            if (
                os.path.exists(self.local_cache_dir)
                and len(os.listdir(self.local_cache_dir)) > 0
            ):
                print("Warning: local cache dir is not empty. Clearing it now.")
                shutil.rmtree(self.local_cache_dir, ignore_errors=True)
            os.makedirs(self.local_cache_dir, exist_ok=True)
        self.features_written = None

    @staticmethod
    def subfolder_name(rank: int, world_size: int) -> str:
        return f"{rank}"

    @staticmethod
    def fileidx2name(file_idx: int) -> str:
        return f"record_{file_idx}.bin"

    def fileidx2path(self, file_idx: int, local_cache_dir: Optional[str] = None):
        """Turn absolute file idx into relative path of the corresponding record file.
        Write to self.local_dir and move.

        Args:
            file_idx (int): _description_
            local_cache_dir (str, optional): The directory to cache the record file. Defaults to None.
        Returns:
            _type_: _description_

        Yields:
            _type_: _description_
        """
        dir = self.recorddir if local_cache_dir is None else local_cache_dir
        rank_id = self.meta_file[file_idx].get("rank_id", -1)
        if rank_id == -1:
            # there is no rank hierarachy
            return os.path.join(dir, self.fileidx2name(file_idx))
        else:
            return os.path.join(
                dir,
                self.subfolder_name(rank_id, self.num_ranks),
                f"record_{self.meta_file[file_idx]['rel_file_idx']}.bin",
            )

    def recordfile_generator(self, frame_generator: Iterator):
        """Ignore the complexity of rank/world size from multi-processing. This method only
        focus on file/frame.

        Args:
            frame_generator (callable): _description_

        Yields:
            _type_: _description_
        """
        try:
            write_buffer = WriterBuffer()
            num_bytes = 0
            self.meta_file = {}
            self.meta_frame = {}
            frame_idx = 0
            file_idx = 0
            for frame in frame_generator:
                if self.features_written is None:
                    self.features_written = [key for key in frame]
                if num_bytes == 0:
                    # new file
                    self.meta_file[file_idx] = {
                        "frame_idx_start": frame_idx,
                        "relative_path": self.fileidx2name(file_idx),
                    }
                    # relative path to the record file does not contain directory of the corresponding seqrecord
                self.meta_frame[frame_idx] = {
                    "file_idx": file_idx,
                    "bytes_offset": num_bytes,
                }
                num_bytes_in_frame = 0
                for feature, data in frame.items():
                    self.meta_frame[frame_idx][feature] = {
                        "is_none": (
                            data.dtype == np.dtype("O") and data == None
                        ),  # this feature is essentially missing, and
                        "dtype": data.dtype,
                        "shape": data.shape,
                        "bytes_offset": num_bytes_in_frame,
                        "nbytes": data.nbytes,
                    }
                    write_buffer.write(data.tobytes())
                    num_bytes_in_frame += data.nbytes

                self.meta_frame[frame_idx]["nbytes"] = num_bytes_in_frame
                frame_idx += 1
                num_bytes += num_bytes_in_frame
                if num_bytes > MAX_RECORDFILE_SIZE:
                    # current file is big enough
                    num_bytes = 0
                    self.meta_file[file_idx]["frame_idx_end"] = frame_idx
                    write_buffer.clear()
                    yield (
                        self.fileidx2path(
                            file_idx, local_cache_dir=self.local_cache_dir
                        ),
                        write_buffer.getvalue(),
                    )
                    file_idx += 1

            if (
                file_idx in self.meta_file
                and self.meta_file[file_idx].get("frame_idx_end", None) is None
            ):
                # there is content left in the write_buffer
                self.meta_file[file_idx]["frame_idx_end"] = frame_idx
                yield (
                    self.fileidx2path(file_idx, self.local_cache_dir),
                    write_buffer.getvalue(),
                )
                file_idx += 1
        finally:
            write_buffer.close()
            self.num_files = file_idx
            self.num_frames = frame_idx

    def put_frame(self, frame_generator: callable, prefetch_buffer_size: int = 5):
        # should be only adding frames here
        # two threads this function keep writing and send them to buffer
        # a separate thread writes the buffer to files as long as the buffer is non-empty
        try:
            prefetch_data = _PrefetchData(
                self.recordfile_generator(frame_generator=frame_generator),
                prefetch_buffer_size,
            )
            thread = threading.Thread(
                target=WSeqRecord.prefetch_thread_worker,
                args=(prefetch_data,),
                daemon=True,
            )
            thread.start()
            file_cache = []
            subprocesses = deque()
            while prefetch_data.run_prefetcher:
                if len(prefetch_data.prefetch_buffer) > 0:
                    (
                        file_path,
                        content,
                    ) = prefetch_data.prefetch_buffer.popleft()
                    with open(file_path, "wb") as f:
                        f.write(content)
                    file_cache.append(file_path)
                    if (
                        self.local_cache_dir is not None
                        and len(file_cache) > FILE_CACHE_SIZE
                    ):
                        # move record files to recorddir in the background
                        subprocesses.append(
                            subprocess.Popen(
                                [
                                    "mv",
                                ]
                                + file_cache
                                + [f"{self.recorddir}/"]
                            )
                        )
                        file_cache = []
                else:
                    # TODO: Calculate sleep interval based on previous availability speed
                    time.sleep(CONSUMER_SLEEP_INTERVAL)
                    # todo: check if poll() is used properly
                    if len(subprocesses) > 0 and subprocesses[0].poll() is not None:
                        subprocesses.popleft()

        finally:
            prefetch_data.run_prefetcher = False
            if thread is not None:
                thread.join()
                thread = None
            if self.local_cache_dir is not None and len(file_cache) > 0:
                subprocesses.append(
                    subprocess.Popen(
                        [
                            "mv",
                        ]
                        + file_cache
                        + [f"{self.recorddir}/"]
                    )
                )
                file_cache = []
            for p in subprocesses:
                p.wait()

    def read_frame(
        self,
        file_desc: Union[io.BufferedReader, BinaryIO],
        metadata_frame: Dict[str, Union[int, dict]],
        features: List[str],
    ) -> Dict[str, np.ndarray]:
        """Given record file descriptor and serialization proto of a single frame, return the
        decoded dictionary(feature->data(np.ndarray)) of the item.

        Args:
            file_desc (io.BufferedReader): python file object of the record file (required by numpy)
            metadata_frame (Dict[str, Any]): dict that contains meta info of a specific frame
            features (List[str]):  features requested for frame
        Returns:
            Dict[str, np.ndarray]: data
        """
        frame = {}
        frame_offset = metadata_frame["bytes_offset"]
        for feature in features:
            frame[feature] = np.memmap(
                file_desc,
                dtype=metadata_frame[feature]["dtype"],
                mode="r",
                offset=frame_offset + metadata_frame[feature]["bytes_offset"],
                shape=metadata_frame[feature]["shape"],
            )
        return frame

    def iterate_frames(
        self, features: List[str]
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Iterate sequentially over frames in the dataset

        Args:
            features (List[str]): a list of feature names requested from frames

        Returns:
            _type_: _description_

        Yields:
            Generator[Dict[str, np.ndarray], None, None]: generates one-frame data
        """
        for file_idx in range(self.num_files):
            file_desc = open(
                os.path.join(self.recorddir, self.meta_file[file_idx]["relative_path"]),
                mode="rb",
            )
            for idx in range(
                self.meta_file[file_idx]["frame_idx_start"],
                self.meta_file[file_idx]["frame_idx_end"],
            ):
                frame = self.read_frame(file_desc, self.meta_frame[idx], features)

                yield {feature: frame[feature] for feature in features}
            file_desc.close()

    # todo: test effect of caching on real data
    def iterate_framepairs(
        self,
        input_features: List[str],
        target_features: List[str],
        max_pred_steps: int,
        filedesc_cache_cap: int = 10,
        frame_cache_cap: int = 20,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Iterate frames over the whole dataset

        # todo: to think about, if we don't shuffle files, then cache based on frame idx is convenient and effective.
        Args:
            input_features [List[str]]: a list of features requested for input
            target_features [List[str]]: a list of features requested for target
            max_pred_steps [int]: maximum number of leap steps for predictive frame
        Yields:
            Generator[Dict[str, np.ndarray], None, None]: data item [feature->data]. All data items are being returned sequentially
        """
        file_manager = FileManager(
            cache_capacity=filedesc_cache_cap,
        )
        # given that, input and target features do not overlap, we only cache target frame
        # LRU might not be suitable, evicting based on idx seems better
        frame_cache = LRUCache(frame_cache_cap)

        for fileidx4input in range(self.num_files):
            filedesc4input = file_manager.open_file(
                file_idx=fileidx4input,
                file_path=os.path.join(
                    self.recorddir, self.meta_file[fileidx4input]["relative_path"]
                ),
            )
            endpoints = (
                self.meta_file[fileidx4input]["frame_idx_start"],
                self.meta_file[fileidx4input]["frame_idx_end"],
            )
            # no target frame to predict for the last frame
            for frameidx4input in range(
                endpoints[0],
                min(endpoints[1], self.num_frames - 1),  # self.num_frames
            ):
                input_frame = self.read_frame(
                    filedesc4input, self.meta_frame[frameidx4input], input_features
                )
                # get the target frame for prediction, both start, stop inclusive
                lookahead_steps = min(
                    random.randint(1, max_pred_steps),
                    self.num_frames - 1 - frameidx4input,
                )
                frameidx4target = frameidx4input + lookahead_steps
                target_frame = frame_cache.get(frameidx4target)
                if target_frame is None:
                    fileidx4target = self.meta_frame[frameidx4target]["file_idx"]
                    filedesc4target = file_manager.open_file(
                        file_idx=fileidx4target,
                        file_path=os.path.join(
                            self.recorddir,
                            self.meta_file[fileidx4target]["relative_path"],
                        ),
                    )
                    target_frame = self.read_frame(
                        filedesc4target,
                        self.meta_frame[frameidx4target],
                        target_features,
                    )
                # colllate input and target frames so that input and target frame are np.ndarray
                input_frame = np.vstack(
                    [input_frame[feature] for feature in input_features]
                )
                target_frame = np.vstack(
                    [target_frame[feature] for feature in target_features]
                )
                yield {
                    "input": input_frame,
                    "target": target_frame,
                    "lookahead_steps": np.asarray(lookahead_steps),
                    "input_features": input_features,
                    "target_features": target_features,
                }
        file_manager.close_all_files()

    def async_iterate_framepairs(
        self,
        input_features: List[str],
        target_features: List[str],
        max_pred_steps: int,
        filedesc_cache_cap: int = 10,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Asyncly read two frames from (possibly) two files.

        Notes:
            No frame cache

        Returns:
            _type_: _description_

        Yields:
            _type_: _description_
        """
        # setup a single event loop for async read
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # file_desc_cache is only used for target frame, since we are iterating for the input frame,
        file_desc_cache = LRUCache(capacity=filedesc_cache_cap)
        # given that, input and target features do not overlap, we only cache target frame
        # LRU might not be suitable, evicting based on idx seems better

        # read two frames using asyn io
        # file cache should only be used for future frames, since base file desc is used continuously
        try:
            for fileidx4input in range(self.num_files):
                filedesc4input = None
                endpoints = (
                    self.meta_file[fileidx4input]["frame_idx_start"],
                    self.meta_file[fileidx4input]["frame_idx_end"],
                )

                # no target frame to predict for the last frame
                for frameidx4input in range(
                    endpoints[0], min(endpoints[1], self.num_frames - 1)
                ):
                    lookahead_steps = min(
                        random.randint(1, max_pred_steps),
                        self.num_frames - 1 - frameidx4input,
                    )
                    frameidx4target = frameidx4input + lookahead_steps
                    fileidx4target = self.meta_frame[frameidx4target]["file_idx"]

                    filedesc4target = file_desc_cache.get(fileidx4target)
                    if filedesc4input is None and filedesc4target is None:
                        # both files need to be opened
                        file_descs = loop.run_until_complete(
                            asyncio.gather(
                                async_open_file(
                                    fileidx4input,
                                    None,
                                    os.path.join(
                                        self.recorddir,
                                        self.meta_file[fileidx4input]["relative_path"],
                                    ),
                                ),
                                async_open_file(
                                    fileidx4target,
                                    file_desc_cache,
                                    os.path.join(
                                        self.recorddir,
                                        self.meta_file[fileidx4target]["relative_path"],
                                    ),
                                ),
                            )
                        )
                        # order of return values are preserved. Ref: https://stackoverflow.com/questions/54668701/asyncio-gather-scheduling-order-guarantee
                        filedesc4input, filedesc4target = file_descs[0], file_descs[1]
                    elif filedesc4input is None:
                        # only need files for input frame
                        file_descs = loop.run_until_complete(
                            async_open_file(
                                fileidx4input,
                                None,
                                os.path.join(
                                    self.recorddir,
                                    self.meta_file[fileidx4input]["relative_path"],
                                ),
                            )
                        )
                        filedesc4input = file_descs
                    elif filedesc4target is None:
                        # only need files for target frame
                        file_descs = loop.run_until_complete(
                            async_open_file(
                                fileidx4target,
                                file_desc_cache,
                                os.path.join(
                                    self.recorddir,
                                    self.meta_file[fileidx4target]["relative_path"],
                                ),
                            )
                        )
                        filedesc4target = file_descs

                    frame_pairs = loop.run_until_complete(
                        asyncio.gather(
                            async_read_frame(
                                filedesc4input,
                                self.meta_frame[frameidx4input],
                                input_features,
                            ),
                            async_read_frame(
                                filedesc4target,
                                self.meta_frame[frameidx4target],
                                target_features,
                            ),
                        )
                    )
                    yield {
                        "input": frame_pairs[0],
                        "target": frame_pairs[1],
                        "lookahead_steps": np.asarray(lookahead_steps),
                        "input_features": input_features,
                        "target_features": target_features,
                    }
                # close file descriptor for

                if filedesc4input is not None:
                    loop.run_until_complete(close_aiofile(filedesc4input))
        finally:
            # close open files
            loop.run_until_complete(close_files_in_cache(file_desc_cache))

            # wrap up async works
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    @staticmethod
    def prefetch_thread_worker(prefetch_data):
        # Lazily import to prevent circular import
        # shc: not sure what this is for?
        from torchdata.dataloader2 import communication

        itr = iter(prefetch_data.source_data_generator)
        stop_iteration = False
        while prefetch_data.run_prefetcher:
            if (
                len(prefetch_data.prefetch_buffer) < prefetch_data.buffer_size
                and not stop_iteration
            ):
                try:
                    item = next(itr)
                    prefetch_data.prefetch_buffer.append(item)
                except StopIteration:
                    stop_iteration = True
                # shc: probably not necessary for now
                except communication.iter.InvalidStateResetRequired:
                    stop_iteration = True
                except communication.iter.TerminateRequired:
                    prefetch_data.run_prefetcher = False
            elif stop_iteration and len(prefetch_data.prefetch_buffer) == 0:
                prefetch_data.run_prefetcher = False
            else:  # Buffer is full, waiting for main thread to consume items
                # TODO: Calculate sleep interval based on previous consumption speed
                time.sleep(PRODUCER_SLEEP_INTERVAL)

    def fetch_framepairs(
        self,
        input_features: List[str],
        target_features: List[str],
        max_pred_steps: int,
        prefetch_buffer_size: int = 10,
    ):

        if prefetch_buffer_size < 1:
            yield from self.iterate_framepairs(
                input_features, target_features, max_pred_steps
            )
        else:
            # ref: https://github.com/pytorch/data/blob/main/torchdata/datapipes/iter/util/prefetcher.py
            # preftech using a separate thread
            try:
                prefetch_data = _PrefetchData(
                    self.iterate_framepairs(
                        input_features, target_features, max_pred_steps
                    ),
                    prefetch_buffer_size,
                )
                thread = threading.Thread(
                    target=WSeqRecord.prefetch_thread_worker,
                    args=(prefetch_data,),
                    daemon=True,
                )
                thread.start()
                while prefetch_data.run_prefetcher:
                    if len(prefetch_data.prefetch_buffer) > 0:
                        yield prefetch_data.prefetch_buffer.popleft()
                    else:
                        # TODO: Calculate sleep interval based on previous availability speed
                        time.sleep(CONSUMER_SLEEP_INTERVAL)
            finally:
                prefetch_data.run_prefetcher = False
                if thread is not None:
                    thread.join()
                    thread = None

    @staticmethod
    def iterate_framepairs_from_files(
        fileidx_generator: Iterator[int],
        filedesc_cache_cap: int = 10,
        frame_cache_cap: int = 20,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        file_manager = FileManager(
            cache_capacity=filedesc_cache_cap,
        )
        frame_cache = LRUCache(frame_cache_cap)

        for record, fileidx4input in fileidx_generator:
            filedesc4input = file_manager.open_file(
                file_idx=fileidx4input,
                file_path=os.path.join(
                    record.recorddir, record.meta_file[fileidx4input]["relative_path"]
                ),
            )
            endpoints = (
                record.meta_file[fileidx4input]["frame_idx_start"],
                record.meta_file[fileidx4input]["frame_idx_end"],
            )
            # no target frame to predict for the last frame
            for frameidx4input in range(
                endpoints[0],
                min(endpoints[1], record.num_frames - 1),  # self.num_frames
            ):
                input_frame = record.read_frame(
                    filedesc4input,
                    record.meta_frame[frameidx4input],
                    record.framereader_args["input_features"],
                )
                # get the target frame for prediction, both start, stop inclusive
                lookahead_steps = min(
                    random.randint(1, record.framereader_args["max_pred_steps"]),
                    record.num_frames - 1 - frameidx4input,
                )
                frameidx4target = frameidx4input + lookahead_steps
                target_frame = frame_cache.get(frameidx4target)
                if target_frame is None:
                    fileidx4target = record.meta_frame[frameidx4target]["file_idx"]
                    filedesc4target = file_manager.open_file(
                        fileidx4target,
                        file_path=os.path.join(
                            record.recorddir,
                            record.meta_file[fileidx4target]["relative_path"],
                        ),
                    )
                    target_frame = record.read_frame(
                        filedesc4target,
                        record.meta_frame[frameidx4target],
                        record.framereader_args["target_features"],
                    )
                # colllate input and target frames so that input and target frame are np.ndarray
                # each feature is a two-dimensional np.ndarray
                # output is channelxheightxwidth
                input_frame = np.stack(
                    [
                        input_frame[feature]
                        for feature in record.framereader_args["input_features"]
                    ],
                    axis=0,
                )
                target_frame = np.stack(
                    [
                        target_frame[feature]
                        for feature in record.framereader_args["target_features"]
                    ],
                    axis=0,
                )
                # print(self.timer.summarize())
                yield {
                    "input": input_frame,
                    "target": target_frame,
                    "lead_times": np.asarray(
                        lookahead_steps * record.framereader_args["hours_per_step"],
                        dtype=input_frame.dtype,
                    ),
                    "meta_data": record.framereader_args,
                }
        file_manager.close_all_files()

    @staticmethod
    def iterate_frames_from_file(
        fileidx_generator: Iterator[int],
        filedesc_cache_cap: int = 10,
        frame_cache_cap: int = 20,
    ):
        """read input/target frames from record files, where input consists of multiple consecutive frames and target frame is one future frame.

        Notes:
            # ! $lead_steps is with respect to the last frame frame in $input_frames
            # ! Assume input target features are the same
        Args:
            fileidx_generator (Iterator[int]): _description_
            filedesc_cache_cap (int, optional): _description_. Defaults to 10.
            frame_cache_cap (int, optional): _description_. Defaults to 20.

        Yields:
            _type_: _description_
        """
        file_manager = FileManager(
            cache_capacity=filedesc_cache_cap,
        )
        frame_cache = LRUCache(frame_cache_cap)
        inputframes_queue = deque()
        for record, fileidx4input in fileidx_generator:
            filedesc4input = file_manager.open_file(
                file_idx=fileidx4input,
                file_path=os.path.join(
                    record.recorddir, record.meta_file[fileidx4input]["relative_path"]
                ),
            )
            endpoints = (
                record.meta_file[fileidx4input]["frame_idx_start"],
                min(
                    record.meta_file[fileidx4input]["frame_idx_end"],
                    record.num_frames - 1,
                    # no target frame left to predict for the last frame
                ),
            )
            # fileidx from generator may not be consecutive
            if len(inputframes_queue) > 0:
                print("testing concatnation ", inputframes_queue[-1][0], endpoints[0])
            if (
                len(inputframes_queue) == 0
                or inputframes_queue[-1][0] + 1 != endpoints[0]
            ):
                # consective files, the input frames queue from last file can be used
                # do we need to worry about memory leak issues here, I guess not?
                print("not concated!")
                inputframes_queue = deque()
            print(f"in the {fileidx4input}th file, {endpoints[0]} to {endpoints[1]}!")
            for frameidx4input in range(endpoints[0], endpoints[1]):
                input_frame = frame_cache.get(frameidx4input)
                if input_frame is None:
                    input_frame = record.read_frame(
                        filedesc4input,
                        record.meta_frame[frameidx4input],
                        record.framereader_args["input_features"],
                    )
                    # no need to cache input frames since it is not likely to be reused
                inputframes_queue.append((frameidx4input, input_frame))
                if (
                    len(inputframes_queue)
                    >= record.framereader_args["num_frames_in_input"]
                ):
                    # ready to eject input/target pairs
                    max_pred_steps = min(
                        record.framereader_args["max_pred_steps"],
                        record.num_frames - 1 - frameidx4input,
                    )
                    # both sides of randint are inclusive
                    lookahead_steps = random.randint(1, max_pred_steps)
                    frameidx4target = frameidx4input + lookahead_steps
                    target_frame = frame_cache.get(frameidx4target)
                    if target_frame is None:
                        fileidx4target = record.meta_frame[frameidx4target]["file_idx"]
                        filedesc4target = file_manager.open_file(
                            fileidx4target,
                            file_path=os.path.join(
                                record.recorddir,
                                record.meta_file[fileidx4target]["relative_path"],
                            ),
                        )
                        target_frame = record.read_frame(
                            filedesc4target,
                            record.meta_frame[frameidx4target],
                            record.framereader_args["target_features"],
                        )
                    # colllate input and target frames so that input and target frame are np.ndarray
                    # each feature is a two-dimensional np.ndarray
                    # input_frames is Lis[cxhxw] with length num_frames_in_input
                    input_frames = [
                        (
                            idx,
                            np.stack(
                                [
                                    frame[feature]
                                    for feature in record.framereader_args[
                                        "input_features"
                                    ]
                                ],
                                axis=0,
                            ),
                        )
                        for idx, frame in inputframes_queue
                    ]

                    # target_frame is cxhxw
                    target_frame = np.stack(
                        [
                            target_frame[feature]
                            for feature in record.framereader_args["target_features"]
                        ],
                        axis=0,
                    )

                    yield {
                        "input": input_frames,  # List[np.ndarray with shape cxhxw]
                        "target": target_frame,  # np.ndarray with shape cxhxw
                        "lead_time": np.asarray(
                            lookahead_steps * record.framereader_args["hours_per_step"],
                            dtype=input_frames[0][1].dtype,
                        ),
                        "meta_data": record.framereader_args,
                    }
                    inputframes_queue.popleft()
        file_manager.close_all_files()

    def dump_record(self, rank: Optional[int] = None) -> None:
        """save attributes of instance of record into a pickled file and yaml file for visual inspection.

        Note:
        saving attribute dict instead of pickled class: pickling class and loading it is a mess because of
        path issues.
        """
        file_name = f"record_{rank}" if rank is not None else "record_all"
        dic = copy.deepcopy(self.__dict__)
        # do not want to pickle a python module
        with open(os.path.join(self.recorddir, f"{file_name}.dict"), mode="wb") as f:
            pickle.dump(dic, file=f)

        # transform some features to make them readable in yaml
        for _, val in dic["meta_frame"].items():
            for feature in dic["features_written"]:
                val[feature]["dtype"] = val[feature]["dtype"].str
                val[feature]["shape"] = list(val[feature]["shape"])
        with open(os.path.join(self.recorddir, f"{file_name}.yaml"), mode="w") as f:
            f.write("# Configs for human inspection only!\n")
            f.write(yaml.dump(dic))

    @classmethod
    def load_record(cls, recorddir: str, rank: Optional[int] = None) -> WSR:
        """return an instance of sequence record from file that stores attributes of record as a
        dict (stored at path).

        Args:
            path (str): path to the file that stores dict of attributes of seqrecord

        Returns:
            WSR: an instance of record
        """

        file_path = os.path.join(
            recorddir, "record_all.dict" if rank is None else f"record_{rank}.dict"
        )
        with open(file_path, mode="rb") as f:
            obj_dict = pickle.load(f)
        obj = cls(
            recorddir=recorddir,
        )
        obj_dict.pop("recorddir", None)
        for key, value in obj_dict.items():
            setattr(obj, key, value)
        return obj

    @classmethod
    def gather_subseqrecords(
        cls,
        recorddir: str,
        world_size: int,
        rank2folder: Optional[Dict[int, str]] = None,
    ) -> WSR:
        # make everything hierarchical to make it consistent
        if rank2folder is None:
            rank2folder = {
                i: cls.subfolder_name(i, world_size) for i in range(world_size)
            }
        sub_records = []
        for i in range(world_size):
            sub_records.append(
                cls.load_record(os.path.join(recorddir, rank2folder[i]), rank=i)
            )

        # combine meta data
        features_written = sub_records[0].features_written

        # meta data on each rank collected data
        meta_rank = {}
        meta_file = {}
        meta_frame = {}
        abs_file_idx = 0
        abs_frame_idx = 0
        for i in range(world_size):
            meta_rank[i] = {
                "file_idx_start": abs_file_idx,
                "file_idx_end": abs_file_idx + sub_records[i].num_files,
                "frame_idx_start": abs_frame_idx,
            }
            for j in range(sub_records[i].num_files):
                meta_file[abs_file_idx] = {
                    "relative_path": os.path.join(
                        rank2folder[i],
                        sub_records[i].meta_file[j]["relative_path"],
                    ),
                    "frame_idx_start": abs_frame_idx,
                }
                for k in range(
                    sub_records[i].meta_file[j]["frame_idx_start"],
                    sub_records[i].meta_file[j]["frame_idx_end"],
                ):
                    meta_frame[abs_frame_idx] = sub_records[i].meta_frame[k]
                    meta_frame[abs_frame_idx]["rel_frame_idx"] = k
                    meta_frame[abs_frame_idx]["file_idx"] = abs_file_idx
                    abs_frame_idx += 1
                meta_file[abs_file_idx]["frame_idx_end"] = abs_frame_idx
                abs_file_idx += 1
            meta_rank[i]["frame_idx_end"] = abs_frame_idx

        record = cls(recorddir)
        record.meta_file = meta_file
        record.meta_frame = meta_frame
        record.meta_rank = meta_rank
        record.features_written = features_written

        record.num_ranks = world_size
        record.num_files = abs_file_idx
        record.num_frames = abs_frame_idx
        return record

    def set_framereader_args(self, args: Dict[str, Any]) -> None:
        self.framereader_args = args


async def async_open_file(
    file_idx: int, file_desc_cache: Optional[LRUCache], file_path: str
):
    file_desc = await aiofiles.open(file_path, "rb")
    if file_desc_cache is not None:
        evicted = file_desc_cache.put(file_idx, file_desc)
        if evicted is not None:
            await evicted.close()
    return file_desc


async def close_files_in_cache(file_desc_cache: LRUCache) -> None:
    for key in file_desc_cache.keys():
        await file_desc_cache.pop(key).close()
    return None


async def close_aiofile(file_desc: io.BufferedReader) -> None:
    await file_desc.close()
    return


# notes: have to use file manager (instead of an lru function since we need to close files)
#        how to verify we are doing async?
# other approaches to be compared with:
#           1. read the whole frame and extract feature data (since reading small pieces of data multiple times is probably slow)
#           2. no async at all
async def async_read_frame(
    file_desc: io.BufferedReader, metadata_frame: dict, features: List[str]
) -> np.ndarray:
    """Given frame metadata and file object that contain frame data, read features from the frame data
    according to features

    Args:
        file_desc (io.BufferedReader): file object that contains the frame data (file object returned by aiofiles) is a subtype of BufferedReader
        metadata_frame (dict): _description_
        features (List[str]): _description_

    Returns:
        np.ndarray: _description_
    """
    await file_desc.seek(metadata_frame["bytes_offset"])
    data_bytes = await file_desc.read(metadata_frame["nbytes"])
    frame = {}
    # read the whole chunk or we read each file separately
    for feature in features:
        # b = file_desc.read(metadata[feature]["nbytes"])
        # array1d = np.frombuffer(
        #     bytes,
        #     dtype=metadata[feature]["dtype"],
        # )
        # frame[feature] = array1d
        # `await` halts `async_read_frame` and gives control back
        start = metadata_frame[feature]["bytes_offset"]
        end = start + metadata_frame[feature]["nbytes"]
        frame[feature] = np.frombuffer(
            data_bytes[start:end],
            dtype=metadata_frame[feature]["dtype"],
        ).reshape(metadata_frame[feature]["shape"])

    frame_array = np.vstack([frame[feature] for feature in features])
    return frame_array
