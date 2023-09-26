import tensorflow as tf
import autoencoder
from autoencoder import SobelFilterLayer, SimpleAutoencoder
import numpy as np 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def parse_function(example_proto):
    # Define the feature description needed to decode the TFRecord
    feature_description = {
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'patch': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the input `tf.train.Example` proto using the dictionary above
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the patch
    depth = parsed_example['depth']
    decoded_patch = tf.io.decode_raw(parsed_example['patch'], tf.float32)
    decoded_patch = tf.reshape(decoded_patch, (64, 64, depth))
    return decoded_patch


def input_target_map_fn(patch):
    return (patch, patch)

file_pattern = "/scratch/fslippe/modis/MOD02/training_data/tf_data/*.tfrecord"
files = tf.data.Dataset.list_files(file_pattern)
num_files = len(tf.io.gfile.glob(file_pattern))

dataset = files.interleave(
    lambda x: tf.data.TFRecordDataset(x)
              .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
              .map(input_target_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
    cycle_length=4,  # number of files read concurrently
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
for (x, y) in dataset.take(5):  # Change 5 to any number of batches you want to check
    print(x.shape, x.dtype, y.shape, y.dtype)
    print(np.mean(x))

# Load your validation data (assuming it's not in TFRecord format)
val_data = np.load("/scratch/fslippe/modis/MOD02/test_data/normalized_testpatches_band(1)_winter20_21.npy")[::2]

# Reload your model (if necessary)
import importlib
importlib.reload(autoencoder)

# Initialize your autoencoder
patch_size = 64
bands = [1]  # You might need to specify the bands here
autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size)

# Set up your optimizer and compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model = autoencoder.model(optimizer=optimizer, loss="combined")

# Train the model on your dataset
batch_size = 32
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
steps_per_epoch = 50000 * num_files // batch_size
model.fit(dataset, validation_data=(val_data, val_data), epochs=100, steps_per_epoch=steps_per_epoch)
