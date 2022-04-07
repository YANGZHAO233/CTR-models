import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import argparse
from utils import create_criteo_dataset
from build_models import build_models
import keras_tuner as kt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def parse_par():
    parser = argparse.ArgumentParser(description='Choose a model')
    parser.add_argument('-n', '--model_name', help='DCN,DCN-V2,DCN-A ...')
    args = parser.parse_args()
    return args

def get_dataset(file_path,val_size=0.2, test_size=0.2, embed_dim=8, batch_size=8):
    # get, clean and split dataset
    feature_columns, (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        create_criteo_dataset(file_path, embed_dim=embed_dim, val_size=val_size, test_size=test_size)
    print('the dataset size train = %s, val = %s, test = %s ' %
          (x_train.shape, x_val.shape, x_test.shape))
    # train
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # vali
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # test
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return feature_columns, train_dataset, val_dataset, test_dataset

def model_function(save_model_path, monitor='val_loss', mode='min'):
    # early stop
    early_stop_fn = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=2,
        verbose=0,
        mode=mode,
        baseline=None,
        restore_best_weights=True
    )

    # save model
    save_model_fn = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_model_path,
        monitor=monitor, verbose=1,
        save_best_only=False, save_weights_only=False, mode=mode,
        save_freq='epoch', options=None
    )
    return early_stop_fn, save_model_fn

def model_builder(hp):
    hp_units = [hp.Int('hidden_units1', min_value=32, max_value=64, step=32),
                hp.Int('hidden_units2', min_value=32, max_value=64, step=32)]
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    if model_name_temp == 'DCN':
        hp_layers = hp.Int('layer_num', min_value=1, max_value=3, step=2)
        hp_cin_size = [32]
        hp_output_dim = hp.Int('output_dim', min_value=1, max_value=8, step=4)
    else:
        hp_cin_size = [hp.Int('cin_size1', min_value=16, max_value=32, step=16),
                       hp.Int('cin_size2', min_value=16, max_value=32, step=16)]
        hp_layers = 3
        hp_output_dim = 1

    params = {'layer_num': hp_layers, 'hidden_units': hp_units, 'output_dim': hp_output_dim, 'cin_size': hp_cin_size}
    model = build_models(feature_columns, params, model_name=model_name)

    model.compile(
        # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Recall(name='recall')],
        loss_weights=None, weighted_metrics=None, run_eagerly=None,
        steps_per_execution=None
    )

    return model

if __name__ == '__main__':
    # parameters
    file_path = '/home/zhaoy/criteo/data/10w/train_10w.txt'
    save_model_path = '/home/zhaoy/criteo/save_models/'
    val_size, test_size = 0.1, 0.1
    embed_dim = 8
    batch_size = 32
    epochs = 2

    # get model name
    args = parse_par()
    model_name = args.model_name
    print('the CTR model: %s' % model_name)

    model_name_temp = model_name.split('_')[0]
    if model_name_temp == 'xDeepFM':
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        # this is for tf.nn.conv1d in xDeepFM

    # save_model_path
    save_model_path = save_model_path + model_name
    if os.path.exists(save_model_path):
        os.system('rm -rf %s' % save_model_path)
    os.system('mkdir -p %s' % save_model_path)

    # get data set
    feature_columns,train_dataset, val_dataset, test_dataset = \
        get_dataset(file_path=file_path,val_size=val_size, test_size=test_size, embed_dim=embed_dim, batch_size=batch_size)

    early_stop_fn, save_model_fn = model_function(save_model_path=save_model_path, monitor='val_loss', mode='min')
                                                    # monitor='val_loss', mode='min'

    # model
    print("hyper-parameters searching ...")
    tuner = kt.Hyperband(model_builder,
                         # objective='val_loss',
                         objective='val_loss',
                         max_epochs=2,
                         factor=3,
                         directory=save_model_path,
                         project_name='searching model of Criteo')

    tuner.search(train_dataset,
                 epochs=epochs,
                 steps_per_epoch=None,
                 validation_data=val_dataset,
                 validation_steps=None,
                 # validation_split=0.2,
                 callbacks=[early_stop_fn]
                 )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    if model_name_temp == 'DCN':
        print(f"""
        The hyperparameter search is complete. 
        The optimal number of units in the first densely-connected layer is {best_hps.get('hidden_units1')} and
        the second  densely-connected layer is {best_hps.get('hidden_units2')},
        the optimal learning rate for the optimizer is {best_hps.get('learning_rate')},
        the output dimension is {best_hps.get('output_dim')},
        the layer number is {best_hps.get('layer_num')}.
        """)
    else:
        print(f"""
                The hyperparameter search is complete. 
                The optimal number of units in the first densely-connected layer is {best_hps.get('hidden_units1')} and
                the second  densely-connected layer is {best_hps.get('hidden_units2')},
                the optimal learning rate for the optimizer is {best_hps.get('learning_rate')},
                the first cin size is {best_hps.get('cin_size1')},
                the second cin size is {best_hps.get('cin_size2')},.
                """)

    model = tuner.hypermodel.build(best_hps)

    print('Training begin...')
    model.fit(
              train_dataset,
              epochs=epochs,
              workers=10,
              use_multiprocessing=True,
              steps_per_epoch=None,
              validation_data=val_dataset,
              validation_steps=None,
              callbacks=[early_stop_fn, save_model_fn]
              )

    print('Evaluating begin...')
    loss, auc, recall = model.evaluate(test_dataset,verbose=0)
    print("Test loss: ", round(loss, 5))
    print("Test AUC: ", round(auc, 5))
    print("Test Recall: ", round(recall, 5))

    model.save(save_model_path)
    model.summary()