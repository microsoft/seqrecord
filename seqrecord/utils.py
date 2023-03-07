from collections import OrderedDict
import io
import os
from typing import Callable, List, Tuple

PRODUCER_SLEEP_INTERVAL = 0.0001  # Interval between buffer fullfilment checks
CONSUMER_SLEEP_INTERVAL = (
    0.0001  # Interval between checking items availablitity in buffer
)


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
    works_remain = works
    loads_per_process = round(works / num_processes)
    for i in range(num_processes):
        if works_remain % (num_processes - i) == 0:
            loads_per_process = works_remain // (num_processes - i)
        else:
            loads_per_process = round(works_remain / (num_processes - i))
        end = min(start + loads_per_process, works)
        ans.append((start, end))
        works_remain -= end - start
        start = end
    return ans


class LRUCache:
    """LRU cache using OrderedDict

    Returns:
        _type_: _description_
    """

    def __init__(self, capacity: int, monitoring: bool = False) -> None:
        self.capacity = capacity
        self.values = OrderedDict()  # Dict[int, io.BufferedReader] = {}
        self.monitoring = monitoring
        if self.monitoring:
            self.hits = 0
            self.misses = 0

    def put(self, key: int, value):
        """add (key, value) to cache following lru policy using ordered dict

        Args:
            ele (_type_): _description_

        Returns:
            evicted element in case there is post-processing needed.
        """
        evicted = None
        if key not in self.values:
            if len(self.values) == self.capacity:
                # popitem() returns key, value pair
                _, evicted = self.values.popitem(last=False)
        else:
            self.values.pop(key)
        self.values[key] = value
        return evicted

    def get(self, key: int):
        """retrieve element from cache

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        res = None
        if key in self.values:
            self.values[key] = self.values.pop(key)
            res = self.values[key]
        if self.monitoring:
            self.hits = self.hits + (res is not None)
            self.misses = self.misses + (res is None)
        return res

    def pop(self, key: int):
        return self.values.pop(key)

    def keys(self):
        return [key for key in self.values.keys()]

    def stats(self) -> str:

        if self.monitoring:
            return f"Cache hitting rate is {self.hits / (self.hits + self.misses)} with overall number of reads {self.hits + self.misses}"
        else:
            return "Monitoring was not enabled for cache, hence no stats available"


class FileManager:
    """manager opening and closing of file descriptors, with a LRU Cache for opened files (remote files, connection)."""

    # todo: add stats for monitoring cache miss and hits

    def __init__(
        self,
        cache_capacity: int,
        monitoring: bool = False,
    ) -> None:
        self.cache = LRUCache(cache_capacity, monitoring)

    def open_file(self, file_idx: int, file_path: str) -> io.BufferedReader:
        f = self.cache.get(file_idx)
        if f is None:
            # todo: make this a variable or function or something fixed
            file_path = file_path
            f = open(file_path, "rb")
            evicted = self.cache.put(key=file_idx, value=f)
            if evicted is not None:
                evicted.close()
        return f

    def close_all_files(self) -> None:
        """Close all open file descriptors in cache"""
        for key in self.cache.keys():
            self.cache.pop(key).close()
        return


class WriterBuffer:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def write(self, buffer) -> None:
        self.buffer.write(buffer)

    def is_empty(self) -> bool:
        return self.buffer.tell() == 0

    def getbuffer(self) -> memoryview:
        return self.buffer.getbuffer()

    def getvalue(self):
        return self.buffer.getvalue()

    def clear(self) -> None:
        self.buffer.flush()
        self.buffer.seek(0)

    def close(self) -> None:
        self.buffer.close()


class TimeTracker:
    def __init__(self) -> None:
        self.time: float = 0.0
        self.nbytes: int = 0

    def add(self, time: float, nbytes: int) -> None:
        self.time += time
        self.nbytes += nbytes

    def summarize(self) -> str:
        return f"Took {self.time} to read {self.nbytes} bytes, with average rate {self.nbytes / self.time} bytes/s"
