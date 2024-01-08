import tensorflow as tf
import autoencoder
from autoencoder import SobelFilterLayer, SimpleAutoencoder
from keras.callbacks import LearningRateScheduler
import numpy as np 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler
import os
import pickle
import sys 
from keras import backend as K

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


class CustomLearningRateScheduler(Callback):
    """Custom learning rate scheduler that runs after ReduceLROnPlateau."""

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Only reduce the lr at the start of epoch 15 or below
        if epoch == 14:
            K.set_value(self.model.optimizer.lr, 1e-4)

class CustomModelCheckpoint(Callback):
    def __init__(self, model, autoencoder, save_folder, model_run_name, save_freq):
        self.model = model
        self.autoencoder = autoencoder
        self.save_folder = save_folder
        self.model_run_name = model_run_name
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            # Save the complete model
            self.model.save(f"{self.save_folder}autoencoder_{self.model_run_name}_epoch_{epoch+1}.h5")
            # Save the individual encoder and decoder
            self.autoencoder.encoder.save(f"{self.save_folder}encoder_{self.model_run_name}_epoch_{epoch+1}.h5")
            self.autoencoder.decoder.save(f"{self.save_folder}decoder_{self.model_run_name}_epoch_{epoch+1}.h5")





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

def scheduler(epoch):
    if epoch < 1:
        return 1e-3
    else:
        return 1e-4


def main():
    #### Define parameters
    bands = [29]  
    if len(sys.argv) < 3:
        patch_size = 128
        last_filter = 64
    else:
        patch_size = int(sys.argv[1])  # Convert first argument to an integer.
        last_filter = int(sys.argv[2])  # Convert second argument to an integer.
        
    if last_filter == 196:
        filters = [16, 64, 128, 196]
    if last_filter == 256:
        filters = [32, 64, 128, 256]
    if last_filter == 128:
        filters = [16, 32, 64, 128]
    elif last_filter == 64:
        filters = [8, 16, 32, 64]
    elif last_filter == 32:
        filters = [4, 8, 16, 32]
    else:
        print("No filters chosen")

    print(f'Patch size is set to: {patch_size}')
    print(f'Filters is set to: {filters}')

    #### Define load and save names
    patch_load_name = "dnb_l95_z50_ps%s_band29" %(patch_size)
    model_run_name = "dnb_l95_z50_ps%s_f%s_1e3_%s-%s" %(patch_size, filters[-1], "201812", "202312")


    #### prepare files
    print(f"/scratch/fslippe/modis/MOD02/training_data/tf_data/{patch_load_name}/normalized_trainingpatches_{patch_load_name}*.tfrecord")
    file_pattern = f"/scratch/fslippe/modis/MOD02/training_data/tf_data/{patch_load_name}/normalized_trainingpatches_{patch_load_name}*.tfrecord"
    files = tf.data.Dataset.list_files(file_pattern)
   
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

    val_data = np.load(f"/scratch/fslippe/modis/MOD02/training_data/tf_data/normalized_valpatches_{patch_load_name}.npy")

    save_folder = f"/uio/hume/student-u37/fslippe/data/models/patch_size{patch_size}/filter{filters[-1]}/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_data))

    # # Batch the dataset
    # val_batch_size = 32  # Adjust batch size according to your GPU memory constraints
    # val_dataset = val_dataset.batch(val_batch_size)

    # # Enable prefetching to improve performance
    # val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    custom_lr_scheduler = CustomLearningRateScheduler()

    lr_schedule = ReduceLROnPlateau(
                                    monitor='val_loss', 
                                    factor=0.1, 
                                    patience=30, 
                                    verbose=1, 
                                    mode='auto',
                                    min_delta=0.00001,
                                    min_lr=1e-5
                                    )


    custom_checkpoint_callback = CustomModelCheckpoint(model, autoencoder, save_folder, model_run_name, save_freq=100)

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True, min_delta=0.000001)
    
    history = model.fit(
            dataset,  
            epochs=1000,
            steps_per_epoch=steps_per_epoch,  
            validation_data=(val_data,val_data),  
            # validation_steps are not needed if your dataset perfectly divides by batch size,
            # if not, you can use the following line:
            #validation_steps=np.ceil(len(val_data) / val_batch_size),
            callbacks=[early_stopping, custom_lr_scheduler, lr_schedule, custom_checkpoint_callback]
    )

    # Save models
    model.save(f"{save_folder}autoencoder_{model_run_name}.h5")
    autoencoder.encoder.save(f"{save_folder}encoder_{model_run_name}.h5")
    autoencoder.decoder.save(f"{save_folder}decoder_{model_run_name}.h5")


    with open(f'{save_folder}/training_history_{model_run_name}.pkl' , 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    main()