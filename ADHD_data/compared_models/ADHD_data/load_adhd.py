import numpy as np
import os

DATA_ROOT = "."

def load_split(data_root, split, combine_classes = True):
    graph_path = os.path.join(data_root, split + "_data.npy")
    label_path = os.path.join(data_root, split + "_combine.npy")
    graph = np.load(graph_path)
    label_age_gender = np.load(label_path)
    label = label_age_gender[:, 0]
    age = label_age_gender[:, 1]
    gender = label_age_gender[:, 2]

    if combine_classes:
        label[label > 0] = 1.

    return graph, age, gender, label

# ==================== pop_GCN ====================
def get_features(matrices):
    N = matrices.shape[0]
    D = matrices.shape[1]
    feat_length = (D * D + D) // 2
    out = np.zeros((N, feat_length))
    for i in range(N):
        out[i, :] = matrices[i, :, :][np.triu_indices(D)]
    return out

def get_graph(age, gender, age_threshold = 0.1):
    N = age.shape[0]
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N): # No self-connections
            if abs(age[i] - age[j]) < age_threshold:
                adj[i, j] += 1
                adj[j, i] += 1

            if gender[i] == gender[j]:
                adj[i, j] += 1
                adj[j, i] += 1

    return adj

def concatenate_matrices(splits):
    all_graph = np.concatenate((splits["train"][0], splits["val"][0], splits["test"][0]), axis = 0)
    all_age = np.concatenate((splits["train"][1], splits["val"][1], splits["test"][1]), axis = 0)
    all_gender = np.concatenate((splits["train"][2], splits["val"][2], splits["test"][2]), axis = 0)
    all_label = np.concatenate((splits["train"][3], splits["val"][3], splits["test"][3]), axis = 0)
    return all_graph, all_age, all_gender, all_label

def get_split_ind(splits):
    train_size = splits["train"][0].shape[0]
    val_size = splits["val"][0].shape[0]
    test_size = splits["test"][0].shape[0]
    train_ind = np.array([ind for ind in range(train_size)])
    val_ind = np.array([ind for ind in range(train_size, train_size+val_size)])
    test_ind = np.array([ind for ind in range(train_size+val_size, train_size+val_size+test_size)])
    return train_ind, val_ind, test_ind

# ==================== AEC ====================
def load_fold(splits):
    # X_train (726, 19900)
    # len(y_train) 726
    X_train = get_features(splits["train"][0])
    y_train = list(splits["train"][3].astype(int))

    X_valid = get_features(splits["val"][0])
    y_valid = list(splits["val"][3].astype(int))

    X_test = get_features(splits["test"][0])
    y_test = list(splits["test"][3].astype(int))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

if __name__ == '__main__':
    splits = {"train" : [],
              "val" : [],
              "test" : []}
    for split in splits.keys():
        splits[split] = load_split(DATA_ROOT, split)
        # print(splits[split][0].shape)

    # ==================== pop_GCN ====================
    train_ind, val_ind, test_ind = get_split_ind(splits)
    all_graph, all_age, all_gender, all_label = concatenate_matrices(splits)
    # print(train_ind, val_ind, test_ind)

    features = get_features(all_graph)
    graph = get_graph(all_age, all_gender, 0.1)
    print(features)
    print(graph)

    # ==================== AEC ====================
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_fold(splits)
    print(X_train, y_train, X_valid, y_valid, X_test, y_test)
