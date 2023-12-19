import tensorflow as tf
import autoencoder
from autoencoder import SobelFilterLayer, SimpleAutoencoder
from keras.callbacks import LearningRateScheduler
import numpy as np 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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

def parse_function(example_proto, patch_size=64):
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
    decoded_patch = tf.reshape(decoded_patch, (patch_size, patch_size, depth))
    return decoded_patch

def input_target_map_fn(patch):
    return (patch, patch)

def scheduler(epoch, lr):
    if epoch < 15:
        return 1e-3
    else:
        return 1e-4


def main():
    #### Define parameters
    patch_size = 128
    bands = [29]  
    filters = [16, 32, 64, 128]
    #filters = [8, 16, 32, 64]
    #filters = [4, 8, 16, 32]

    #### Define load and save names
    patch_load_name = "dnb_l95_z50_ps%s_(29)_%s-%s" %(patch_size, "cao_months_20181216", "20231215")
    model_run_name = "dnb_l95_z50_ps%s_f%s_%s-%s" %(patch_size, filters[-1], "201812", "202312")


    #### prepare files
    print(f"/scratch/fslippe/modis/MOD02/training_data/tf_data/normalized_trainingpatches_{patch_load_name}*.tfrecord")
    file_pattern = f"/scratch/fslippe/modis/MOD02/training_data/tf_data/normalized_trainingpatches_{patch_load_name}*.tfrecord"
    files = tf.data.Dataset.list_files(file_pattern)
    val_data = np.load(f"/scratch/fslippe/modis/MOD02/training_data/tf_data/normalized_valpatches_{patch_load_name}.npy")
    num_files = len(tf.io.gfile.glob(file_pattern))
    print("Number of tfrecord files:", num_files)

    dataset = files.interleave(
    lambda x: tf.data.TFRecordDataset(x)
            .map(lambda item: parse_function(item, patch_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(input_target_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
    cycle_length=20,  # number of files read concurrently
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Set up model 
    autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size, filters=filters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model = autoencoder.model(optimizer=optimizer, loss="combined")

    # Train the model on your dataset
    batch_size = 32 
    total_records = sum(1 for _ in dataset) #534460
    total_records -= total_records % batch_size 
    print(total_records)
    buffer_size = total_records #patches_per_file * num_files
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    steps_per_epoch = total_records // batch_size #patches_per_file * num_files // batch_size

    lr_schedule = ReduceLROnPlateau(
                                    monitor='val_loss', 
                                    factor=0.1, 
                                    patience=20, 
                                    verbose=1, 
                                    mode='auto',
                                    min_lr=1e-5
                                    )


    save_folder = f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{filter_size}/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    

    #lr_schedule = LearningRateScheduler(scheduler, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    history = model.fit(dataset,
                        validation_data=(val_data, val_data),
                        epochs=500,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[early_stopping, lr_schedule])

    # Save models
    model.save(f"{save_folder}autoencoder_{model_run_name}.h5")
    autoencoder.encoder.save(f"{save_folder}encoder_{model_run_name}.h5")
    autoencoder.decoder.save(f"{save_folder}decoder_{model_run_name}.h5")

    import pickle

    with open(f'{save_folder}/training_history_{model_run_name}.pkl' , 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    main()