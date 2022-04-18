# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Imagenet dataset reader."""

from collections import namedtuple
from typing import Optional, Sequence, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

IMAGE_SIZE = 224
IMAGE_PADDING_FOR_CROP = 32


DATASET_NAME_TO_TFDS_DATASET = {
    'imagenet': 'imagenet2012:5.*.*',
    'places365': 'places365_small',
}

DATASET_NUM_CLASSES = {
    'imagenet': 1000,
    'places365': 365,
}

DATASET_NUM_TRAIN_EXAMPLES = {
    'imagenet': 12811,
    'places365': 1803460,
}

DATASET_NUM_EVAL_EXAMPLES = {
    'imagenet': 50000,
    'places365': 36500,
}

DATASET_EVAL_TFDS_SPLIT = {
    'imagenet': tfds.Split.VALIDATION,
    'places365': tfds.Split.VALIDATION,
}


DatasetSplit = namedtuple('DatasetSplit',
                          ['tfds_dataset',
                           'tfds_split',
                           'num_examples',
                           'num_classes'])


def get_train_dataset_split(dataset_name: str) -> DatasetSplit:
  return DatasetSplit(
      DATASET_NAME_TO_TFDS_DATASET[dataset_name],
      tfds.Split.TRAIN,
      DATASET_NUM_TRAIN_EXAMPLES[dataset_name],
      DATASET_NUM_CLASSES[dataset_name])


def get_eval_dataset_split(dataset_name: str) -> DatasetSplit:
  return DatasetSplit(
      DATASET_NAME_TO_TFDS_DATASET[dataset_name],
      DATASET_EVAL_TFDS_SPLIT[dataset_name],
      DATASET_NUM_EVAL_EXAMPLES[dataset_name],
      DATASET_NUM_CLASSES[dataset_name])


def _shard(split: DatasetSplit, shard_index: int, num_shards: int) -> Tuple[int, int]:
    """Returns [start, end) for the given shard index."""
    assert shard_index < num_shards
    arange = np.arange(split.num_examples)
    shard_range = np.array_split(arange, num_shards)[shard_index]
    start, end = shard_range[0], (shard_range[-1] + 1)
    return start, end


def load(split: DatasetSplit, is_training: bool, batch_dims: Sequence[int], tfds_data_dir: Optional[str] = None):
    """Loads the given split of the dataset."""
    if is_training:
        start, end = _shard(split, jax.host_id(), jax.host_count())
    else:
        start, end = _shard(split, 0, 1)
    ds = tfds.load('imagenet2012_subset', split='train', shuffle_files=True)
    total_batch_size = np.prod(batch_dims)

    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    if is_training:
        options.experimental_deterministic = False
    ds = ds.with_options(options)

    if is_training:
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

    def preprocess(example):
        image=tf.image.resize_with_crop_or_pad(example['image'], 224, 224)
        image = tf.transpose(image, (2, 0, 1))  # transpose HWC image to CHW format
        image = tf.cast(image, tf.float32)
        label = tf.cast(example['label'], tf.int32)
        return {'images': image, 'labels': label}

    ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=(is_training == False))

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    yield from tfds.as_numpy(ds)


def normalize_image_for_view(image):
    """Normalizes dataset image into the format for viewing."""
    image *= np.reshape(STDDEV_RGB, (3, 1, 1))
    image += np.reshape(MEAN_RGB, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0))
    return image.clip(0, 255).round().astype('uint8')


