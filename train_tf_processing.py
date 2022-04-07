import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from build_models import build_models
from train import model_function


def get_batch_dataset(file_path, batch_size, header, is_shuffle=False):
    raw_dataset = tf.data.experimental.make_csv_dataset(
        file_path, batch_size=batch_size,
        column_names=header, na_value='', label_name=None,
        num_epochs=1, header=False, field_delim=',',
        shuffle=is_shuffle, ignore_errors=True,
        shuffle_seed=batch_size*2+1,
        prefetch_buffer_size=None,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )
    return raw_dataset

def label_feat(feat, label_name):
    label = feat.pop(label_name)
    return feat, label

def get_sparse_dict():
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    sparse_features_cnt = [104, 383, 718, 965, 42, 9, 2000, 67, 3, 1464, 1682, 741,
                           1498, 24, 1368, 841, 10, 899, 320, 4, 775, 10, 14, 952,
                           39, 714]

    sparse_feat = {}
    for i in range(len(sparse_features)):
        sparse_feat[sparse_features[i]] = range(sparse_features_cnt[i])
    return sparse_feat

def build_feature_columns(dense_feat_header, embed_dim):
    dense_feature_columns = []
    sparse_feature_columns = []
    for header in dense_feat_header:
        dense_feature_columns.append(tf.feature_column.numeric_column(header)) # float

    sparse_feat_dict = get_sparse_dict()
    for feature, vcab in sparse_feat_dict.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature,vocabulary_list=vcab)
        sparse_feature_columns.append(tf.feature_column.embedding_column(cat_col, dimension=embed_dim))

    return dense_feature_columns + sparse_feature_columns

if __name__ == '__main__':
    # parameters
    file_path = '/home/zhaoy/criteo/data/10w'
    save_model_path = '/home/zhaoy/criteo/save_models/DCN_tf'
    batch_size = 32
    embed_dim = 16
    epochs = 3
    learning_rate = 0.001
    params = {'layer_num': 6, 'hidden_units': [128, 64], 'output_dim': 1}

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    features_list = ['label'] + dense_features + sparse_features

    # get dataset
    train_dataset =  get_batch_dataset(file_path + '/train_8w.txt_PostProcessing',
                                       batch_size=batch_size,
                                       header=features_list,
                                       is_shuffle=True)
    val_dataset =  get_batch_dataset(file_path + '/vail_1w.txt_PostProcessing',
                                       batch_size=batch_size,
                                       header=features_list,
                                       is_shuffle=True)
    test_dataset =  get_batch_dataset(file_path + '/test_1w.txt_PostProcessing',
                                       batch_size=batch_size,
                                       header=features_list,
                                       is_shuffle=True)

    # separate label and feature
    train_dataset = train_dataset.map(lambda ele: label_feat(ele, features_list[0]),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(lambda ele: label_feat(ele, features_list[0]),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(lambda ele: label_feat(ele, features_list[0]),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print('raw data is completed')

    #embedding
    feature_columns = build_feature_columns(dense_features, embed_dim=embed_dim)
    print('embedding is complete')

    # model build
    if os.path.exists(save_model_path):
        os.system('rm -rf %s' % save_model_path)
    os.system('mkdir -p %s' % save_model_path)

    early_stop_fn, save_model_fn = model_function(save_model_path=save_model_path, monitor='val_auc', mode='max')
    model = build_models(feature_columns, params, model_name='DCN_tf')
    model.compile(
        # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Recall(name='recall')],
        loss_weights=None, weighted_metrics=None, run_eagerly=None,
        steps_per_execution=None
    )

    print("Embedding dimension: %d\n"
          "Batch size: %d\n"
          "Cross layer number: %d\n"
          "Hidden units: %s\n"
          "Output dimension of Dense layer: %d\n"
          "Total epochs: %d" % (
          embed_dim, batch_size, params['layer_num'], params['hidden_units'], params['output_dim'], epochs))

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
    loss, auc, recall = model.evaluate(test_dataset, verbose=0)
    print("Test loss: ", round(loss, 5))
    print("Test AUC: ", round(auc, 5))
    print("Test Recall: ", round(recall, 5))

    model.save(save_model_path)
    model.summary()




