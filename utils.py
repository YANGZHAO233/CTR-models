import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_DataFrame(file, file_size=10000, is_sep=False, names=None):
    mylist = []
    if not is_sep:
        for chunk in pd.read_csv(file, chunksize=file_size):
            mylist.append(chunk)
    else:
        for chunk in pd.read_csv(file, chunksize=file_size, sep='\t', names=names):
            mylist.append(chunk)
    temp_df = pd.concat(mylist, axis=0)
    del mylist
    return temp_df

def block_shot(floor, root):
    def f(x):
        if x < floor:
            x = floor
        elif x > root:
            x = root
        return x
    return f

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def create_criteo_dataset(file_path, embed_dim=8, val_size=0.2,test_size=0.2):
    # feature names of each column
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    features_list = ['label'] + dense_features + sparse_features

    # get data
    df = get_DataFrame(file_path, file_size=10000, is_sep=True, names=features_list)

    # data cleaning
    for col in dense_features:
        df[col] = df[col].fillna(df[col].median())       # fill none in dense_features

    for col in dense_features:
        min_value = df[col].min()
        df[col] = df[col].map(block_shot(floor=min_value, root=df[col].quantile(0.95)))
                                                         # clean outliers using block shot method
    for col in dense_features:
        df.loc[df[col] > 2, col] = df[col].apply(np.log2)
                                                         # transfer dataset to log2
    for col in sparse_features:
        df[col] = df[col].fillna('unknown1')             # fill none of sparse features

    arr = []
    for col in sparse_features:
        value_counts = df[col].value_counts()
        for i in range(len(value_counts)):
            if value_counts[i] >= 10:
                arr.append(value_counts.index[i])        # keep the categories with frequency larger than 10
            else:
                df[col] = np.where(df[col].isin(arr), df[col], 'unknown2')
                                                         # replace frequency less than 10 with 'unknown2'
                arr = []
                break

    # LabelEncoding
    for col in sparse_features:
        df[col] = LabelEncoder().fit_transform(df[col]).astype(int)

    # embedding
    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                    [[sparseFeature(feat, df[feat].nunique(), embed_dim) for feat in sparse_features]]

    # label and features
    x = df.drop(['label'], axis=1).values
    y = df['label']

    # temp = []
    # for col in sparse_features:
    #     value_counts = df[col].value_counts()
    #     temp.append(len(value_counts))
    # print(temp)

    # split dataset to train, validation and test
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)

    data_size = x.shape[0]
    test_size_temp = data_size * test_size      # 100 * 0.2 = 20
    data_size_temp = x_train.shape[0]           # 80
    test_size = test_size_temp / data_size_temp # 20 / 80 = 0.25
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)

    # save data to csv
    # df.to_csv(file_path + '_PostProcessing', index=False)
    return feature_columns, (x_train, y_train), (x_val, y_val), (x_test, y_test)

