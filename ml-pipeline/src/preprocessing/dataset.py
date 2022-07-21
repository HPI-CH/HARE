import numpy as np
import tensorflow as tf


def reduce_multiwindow_to_window(dataset: tf.data.Dataset, *, n_windows: int):
    def fn(windows, labels):
        window_dataset = windows.batch(n_windows)
        label_dataset = labels.skip(n_windows - 1).take(1)
        return tf.data.Dataset.zip((window_dataset, label_dataset))

    window_count = tf.data.experimental.cardinality(dataset)
    dataset = dataset.window(n_windows, shift=1, drop_remainder=True).flat_map(fn)
    return dataset.apply(tf.data.experimental.assert_cardinality(window_count - n_windows + 1))


def flatten(windowed_dataset: tf.data.Dataset) -> tf.data.Dataset:
    cardinality = windowed_dataset.reduce(np.int64(0), lambda x, ds: x + tf.data.experimental.cardinality(ds[0]))

    def fn(windows, labels):
        return tf.data.Dataset.zip((windows, labels))

    return windowed_dataset.flat_map(fn).apply(tf.data.experimental.assert_cardinality(cardinality))
