# SeqRecord

## Basic Idea

We focus on video-like datasets that consist of data-frames collected along trajectories of varying lengths. And each data-frame contains data measurements from different sources (e.g., `image_left` and `image_right`) and modalities (e.g., `Camera` and `LiDAR`). During training, we usually need samples of consecutive data-frames of fixed length, i.e., video-segment. `SeqRecord` provides efficient sequential access to video-segments for video-like datasets.

## Basic Usage

Given existing dataset, we first need to transform the dataset into binary format by sequentially calling `SeqRecord.write_item(item: Dict[str, np.ndarray], is_seq_start:bool)`:
* `item: Dict[str, np.ndarray]` contains all data collected at each frame
* `is_seq_start: bool` denotes if the current data-frame is the beginning of a sequence

`example/record_dataset.py` provides a simple example, where we first create a `SeqRecord` object, and iteratively call `write_item`. When creating the `SeqRecord` object, three parameters are specified:
1. `recorddir: str`: the directory that formatted data is stored
2. `features: Dict[str, str]`: mapping between each input (in data-frame) and its modality class, e.g., `{image_left:RGBImage}`
3. `pretransform_module_path:str`: When writing existing dataset into binary format, certain data-transforms can be applied to inputs in data-frame without regrets while saving later processing time (`pretransform`). `pretransform_module_path` specifies the path to the module where Modality classes with pretransform are defined. 


When accessed for training or evaluation, specific length of video-segment, and inputs needed from each data-frame need to be provided to create a torch `datapipe` that generates video-segment from the dataset. See `example/dataset_datapipes.py` for more details.
