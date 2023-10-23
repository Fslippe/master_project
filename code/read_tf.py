import tensorflow as tf
import autoencoder
from autoencoder import SobelFilterLayer, SimpleAutoencoder
from keras.callbacks import LearningRateScheduler
import numpy as np 
from tensorflow.keras.callbacks import EarlyStopping

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

def scheduler(epoch, lr):
    if epoch < 15:
        return 1e-3
    else:
        return 1e-4


def main():
    # Define LearningRateScheduler callback
    model_run_name = "dnb_l90_z50_(29)_%s-%s" %("cao_months_202012", "202111")
    print(f"/scratch/fslippe/modis/MOD02/training_data/tf_data/normalized_trainingpatches_{model_run_name}*.tfrecord")
    file_pattern = f"/scratch/fslippe/modis/MOD02/training_data/tf_data/normalized_trainingpatches_{model_run_name}*.tfrecord"
    files = tf.data.Dataset.list_files(file_pattern)
    num_files = len(tf.io.gfile.glob(file_pattern))

    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x)
                .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .map(input_target_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
        cycle_length=20,  # number of files read concurrently
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    for (x, y) in dataset.take(5):  # Change 5 to any number of batches you want to check
        print(x.shape, x.dtype, y.shape, y.dtype)
        print(np.mean(x))

    max_value = float('-inf')
    for (x, y) in dataset:
        current_max = np.max(x.numpy())
        if current_max > max_value:
            max_value = current_max

    print("Max Value in the dataset:", max_value)
    # Load your validation data (assuming it's not in TFRecord format)
    val_data = np.load(f"/scratch/fslippe/modis/MOD02/training_data/tf_data/normalized_valpatches_{model_run_name}.npy")

    # Reload your model (if necessary)

    # Initialize your autoencoder
    patch_size = 64
    bands = [1]  # You might need to specify the bands here
    filters = [8, 16, 32, 64]
    autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size, filters=filters)

    # Set up your optimizer and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model = autoencoder.model(optimizer=optimizer, loss="combined")

    # Train the model on your dataset
    batch_size = 32 
    total_records = sum(1 for _ in dataset) #534460
    print(total_records)
    buffer_size = total_records#patches_per_file * num_files
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    steps_per_epoch = total_records // batch_size #patches_per_file * num_files // batch_size
    lr_schedule = LearningRateScheduler(scheduler, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    history = model.fit(dataset, validation_data=(val_data, val_data), epochs=500, steps_per_epoch=steps_per_epoch, callbacks=[early_stopping, lr_schedule])

    model_run_name = "scheduler_250k_dnb_l90_z50_f%s_(29)_%s-%s" %("cao_months_202012", "202111", filters[-1])

    model.save("/uio/hume/student-u37/fslippe/data/models/autoencoder_%s" %(model_run_name))
    autoencoder.encoder.save("/uio/hume/student-u37/fslippe/data/models/encoder_%s" %(model_run_name))
    autoencoder.decoder.save("/uio/hume/student-u37/fslippe/data/models/decoder_%s" %(model_run_name))

    import pickle
    with open('training_history%s.pkl' %(model_run_name), 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    main()