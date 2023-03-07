"""A package for decoding and encoding each item in data file."""

import collections
import io
import os
import pickle
from typing import (
    Any,
    BinaryIO,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Deque,
)
import numpy as np
import yaml
from seqrecord.utils import WriterBuffer
import copy
from collections import deque
import time
from seqrecord.utils import PRODUCER_SLEEP_INTERVAL, CONSUMER_SLEEP_INTERVAL
import threading

MAX_RECORDFILE_SIZE = 1e9  # 1e8, 100 mb, maximum size of a single record file

RSR = TypeVar("RSR", bound="RSeqRecord")


def recordfileidx2path(recorddir, recordfile_idx: int) -> str:
    return os.path.join(recorddir, f"records_{recordfile_idx}.bin")


# todo: add metadata for each episode


class RSeqRecord:
    """A serialization protocal that stores sequences of episodes into record files, while provides metadata of each episode/frame
    to read segments from dataset."""

    def __init__(
        self,
        recorddir: str,
    ) -> None:
        # folder that stores data in a separate directory (subfolder)
        self.recorddir: str = recorddir
        os.makedirs(self.recorddir, exist_ok=True)

        self.features_written = None
        self.num_bytes: int = 0  # number of bytes written into current record file
        self.file_idx: int = 0  # number of record file created for dataset
        # track the idx endpoints for each record file, [[start_idx, end_idx]], both are inclusive
        self.idx_range_of_files: List[Tuple[int, int]] = []
        # file object for current record file
        self.file_desc: Optional[BinaryIO] = None
        # serialization proto info of each data item
        self.metadata: Dict[int, dict] = {}
        # index of current data item to be processed
        self.frame_idx: int = 0

        # a cache dict that stores protocal info for each (segment_len, sub features)
        self.metadata_segment_cache: Dict[str, dict] = {}
        self.write_buffer = WriterBuffer()

    def get_recordfiles(self) -> List[str]:
        return [self.recordfile_idx_to_path(i) for i in range(self.file_idx)]

    def write_item(
        self,
        frame: Dict[str, np.ndarray],
        is_seq_start: bool,
    ) -> None:
        """write one item data dict(feature->np.ndarray) into bytes and write encoded bytes into
        current record files.

        Args:
            item (Dict[str, np.ndarray]): feature to data (np.ndarray)
            is_seq_start (bool): denote if the item is the beginning of a sequence
        """
        if self.features_written is None:
            self.features_written = [key for key in frame]
        if is_seq_start:
            self.seq_start()

        # get file name and start position for item
        self.metadata[self.frame_idx] = {
            "frame_idx": self.frame_idx,
            "file_idx": self.file_idx,
            "bytes_offset": self.num_bytes,
            "is_seq_start": is_seq_start,
        }
        num_bytes_in_frame = 0
        for feature, data in frame.items():
            self.metadata[self.frame_idx][feature] = {
                "is_none": (
                    data.dtype == np.dtype("O") and data == None
                ),  # this feature is essentially missing, and
                "dtype": data.dtype,
                "shape": data.shape,
                "bytes_offset": num_bytes_in_frame,
                "nbytes": data.nbytes,
            }
            self.write_buffer.write(data.tobytes())
            num_bytes_in_frame += data.nbytes

        self.num_bytes += num_bytes_in_frame
        self.frame_idx += 1

        return

    def seq_start(self) -> None:
        """Notify the record that a new sequence is being written, let the record decide if we need
        a new record file to write into.

        Two cases we need to open new file:
            1. we currently do not have record file to write into
            2. current file size is big enough (larger than MAX_RECORDFILE_SIZE)
        """
        if self.num_bytes > MAX_RECORDFILE_SIZE:
            # current record file big enough
            self.num_bytes = 0
            self.file_idx += 1
            self.idx_range_of_files[-1].append(self.frame_idx - 1)
            self.idx_range_of_files.append([self.frame_idx])
            self.file_desc.write(self.write_buffer.getbuffer())
            self.file_desc.flush()
            self.file_desc.close()
            self.write_buffer.clear()
            self.file_desc = open(
                recordfileidx2path(self.recorddir, self.file_idx),
                mode="wb",
            )
        elif self.file_desc == None:
            # no opened record file to write into
            self.idx_range_of_files.append([self.frame_idx])
            self.file_desc = open(
                recordfileidx2path(self.recorddir, self.file_idx), mode="wb"
            )

    def read_frame(
        self,
        file_desc: Union[io.BufferedReader, BinaryIO],
        metadata_frame: Dict[str, Union[int, dict]],
        features: List[str],
    ) -> Dict[str, np.ndarray]:
        """Given record file descriptor and serialization proto of a single data item, return the
        decoded dictionary(feature->data(np.ndarray)) of the item.

        Args:
            recordfile_desc (io.BufferedReader): python file object of the record file (required by numpy)
            itemproto (Dict[str, Any]): dict that contains protocal info of a specific data item

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
        # * do we need to close the memmap?
        return frame

    def read_frame_frombuffer(
        self,
        file_desc: Union[io.BufferedReader, BinaryIO],
        metadata_frame: Dict[str, Union[int, dict]],
        features: List[str],
    ) -> Dict[str, np.ndarray]:
        """Given record file descriptor and serialization proto of a single data item, return the
        decoded dictionary(feature->data(np.ndarray)) of the item, where decoding is done by
        np.frombuffer()

        Args:
            recordfile_desc (io.BufferedReader): python file object of the record file (required by numpy)
            itemproto (Dict[str, Any]): dict that contains protocal info of a specific data item

        Returns:
            Dict[str, np.ndarray]: data
        """
        frame = {}
        file_desc.seek(metadata_frame["bytes_offset"])
        for feature in features:
            bytes = file_desc.read(metadata_frame[feature]["nbytes"])
            array1d = np.frombuffer(
                bytes,
                dtype=metadata_frame[feature]["dtype"],
            )
            frame[feature] = array1d.reshape(metadata_frame[feature]["shape"])
        return frame

    def read_frames(
        self, features: List[str]
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Given that the dataset has been recored, decode the record sequentially, each time
        returning a dict that contains the data item.

        Args:
            features [List[str]]: a list of features requested to read from item
            shuffle_recordfile: bool: if we shuffle at the record file level when reading items in the record
        Yields:
            Generator[Dict[str, np.ndarray], None, None]: data item [feature->data]. All data items are being returned sequentially
        """
        recordfiles = list(range(self.file_idx))
        for i in recordfiles:
            recordfile_path = recordfileidx2path(self.recorddir, i)
            endpoints = self.idx_range_of_files[i]
            with open(recordfile_path, mode="rb") as f:
                for idx in range(endpoints[0], endpoints[1] + 1):
                    item = self.read_frame(f, self.metadata[idx], features)
                    yield {feature: item[feature] for feature in features}

    def get_metadata4segment(
        self, segment_len: int, sub_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a protocal for reading segments from records. Each data item of segment should
        contain all features in sub_features.

        Note:
        Only call this function when record has scanned all data in dataset, and record has valid attributes: rootdir, recordfile_idx

        Args:
            segment_len (int): length of segment we are reading, 1< segment_len < sequence length
            sub_features: (Optional[List[str]]): features (modalities data) we need for each data item in segment to contain. If it is None,
            then we read all features.

        Returns:
            Dict[str, Any]: protocal needed for reading segments from data
        """

        def has_sub_features(itemproto: Dict[str, Any]) -> bool:
            return all(not itemproto[feature]["is_none"] for feature in sub_features)

        def update_segmentproto(item_idx: int, is_segment_start: bool) -> None:
            if is_segment_start:
                head4segment.append(item_idx)
            recordfile_idx = self.metadata[item_idx]["file_idx"]
            file2segment_items[recordfile_idx].append((is_segment_start, item_idx))
            return

        if sub_features is None:
            sub_features = list(self.features_written)
        else:
            assert all(
                feature in self.features_written for feature in sub_features
            ), "Unknow features requested"
        cache_key = str(segment_len) + "#" + "#".join(sorted(sub_features))
        if cache_key in self.metadata_segment_cache:
            return self.metadata_segment_cache[cache_key]
        head4segment: List[int] = []
        file2segment_items: dict[int, List[Tuple[bool, int]]] = collections.defaultdict(
            list
        )
        q = collections.deque()
        q_has_seg_tail = False  # indicates if the elements currently in queue are tail of some segment
        for idx in range(self.frame_idx):
            itemproto = self.metadata[idx]
            if (not has_sub_features(itemproto)) or (itemproto["is_seq_start"]):
                # new seq start
                while q:
                    if q_has_seg_tail:
                        update_segmentproto(q.popleft(), is_segment_start=False)
                    else:
                        q.popleft()
                q_has_seg_tail = False
                if has_sub_features(itemproto):
                    # a valid start of sequence
                    q.append(idx)
            else:
                q.append(idx)
                if len(q) == segment_len:
                    # claim: elements in the queue must be from the same sequence
                    update_segmentproto(q.popleft(), is_segment_start=True)
                    q_has_seg_tail = True

        if q and q_has_seg_tail:
            # front element in queue is need as last element of some segment
            update_segmentproto(q.popleft(), is_segment_start=False)

        # 1. new seq (including broken) added before queue pops out
        #       the remaining elements in queue are completely useless
        # 2. new seq (including broken) added after queue has popped out
        #       the remaining elements are not start of segment but are tails of some segment
        self.metadata_segment_cache[cache_key] = {
            "segment_len": segment_len,
            "features": sub_features,
            "head4segment": head4segment,
            "file2segment_items": file2segment_items,
        }
        return self.metadata_segment_cache[cache_key]

    def read_segments(self, segment_proto: dict):
        """Iterate through the whole records and return segments sequential.

        Yields:
            segment_proto: info on in given segment_len and features
        """
        segment_len = segment_proto["segment_len"]
        recordfile_ids = list(segment_proto["file2segment_items"].keys())
        for recordfile_idx in recordfile_ids:
            item_list = segment_proto["file2segment_items"][recordfile_idx]
            recordfile_path = recordfileidx2path(self.recorddir, recordfile_idx)
            q = collections.deque()
            with open(recordfile_path, mode="rb") as f:
                for is_segment_start, item_idx in item_list:
                    q.append(
                        (
                            is_segment_start,
                            self.read_frame(
                                f, self.metadata[item_idx], segment_proto["features"]
                            ),
                        )
                    )
                    while not q[0][0]:
                        q.popleft()
                    if len(q) == segment_len:
                        yield self.collate_items(q)
                        q.popleft()

    def read_one_segment(
        self,
        segment_len: int,
        head_idx: int,
    ) -> Dict[str, List[np.ndarray]]:
        """Read a segment (of lenght segment_len) starting from the item index being head_idx.

        Args:
            segment_len (int): length of segment we need to generate
            head_idx (int): item_idx of the head of the segment to be read.

        Returns:
            Dict[str, np.ndarray]: segment data
        """
        recordfile_path = recordfileidx2path(
            self.recorddir, self.metadata[head_idx]["recordfile_idx"]
        )
        q = []
        with open(recordfile_path, mode="rb") as f:
            for idx in range(head_idx, head_idx + segment_len):
                q.append(
                    (
                        idx == head_idx,
                        self.read_frame_frombuffer(f, self.metadata[idx]),
                    )
                )
        return self.collate_items(q)

    def collate_items(
        self, q: Sequence[Tuple[bool, dict]]
    ) -> Dict[str, List[np.ndarray]]:
        segment = {}
        features = q[0][1].keys()
        for feature in features:
            segment[feature] = [item[feature] for _, item in q]
        return segment

    def close_recordfile(self):
        """Close opened file descriptor!

        This needs to be called when finishes scanning over the dataset.
        """
        self.idx_range_of_files[-1].append(self.frame_idx - 1)
        self.file_desc.write(self.write_buffer.getbuffer())
        self.write_buffer.close()
        self.write_buffer = None
        self.file_desc.flush()
        self.file_desc.close()
        self.file_idx += 1
        self.file_desc = None

    def dump(self) -> None:
        """save attributes of instance of record into a file.

        Note:
        saving attribute dict instead of pickled class: pickling class and loading it is a mess because of
        path issues.
        """
        dic = copy.deepcopy(self.__dict__)
        with open(os.path.join(self.recorddir, "record.dict"), mode="wb") as f:
            pickle.dump(dic, file=f)

        # save some attributes of the seqrecord to yaml for human inspection
        dic["metadata_segment_cache"] = None
        for key, val in dic["metadata"].items():
            for feature in dic["features_written"]:
                val[feature]["dtype"] = val[feature]["dtype"].str
                val[feature]["shape"] = list(val[feature]["shape"])
        with open(os.path.join(self.recorddir, "record_dict.yaml"), mode="w") as f:
            f.write("# Configs for human inspection only!\n")
            f.write(yaml.dump(dic))

    @classmethod
    def load_record_from_dict(cls, recorddir: str) -> RSR:
        """return an instance of sequence record from file that stores attributes of record as a
        dict (stored at path).

        Args:
            path (str): path to the file that stores dict of attributes of seqrecord

        Returns:
            SR: an instance of record
        """

        file_path = os.path.join(recorddir, "record.dict")
        with open(file_path, mode="rb") as f:
            obj_dict = pickle.load(f)
        obj = cls(
            recorddir=recorddir,
        )
        obj_dict.pop("recorddir", None)
        for key, value in obj_dict.items():
            setattr(obj, key, value)
        return obj
