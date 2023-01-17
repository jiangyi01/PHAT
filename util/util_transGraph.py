import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
import torch


class CONFIG(object):
    """docstring for CONFIG"""

    def __init__(self):
        super(CONFIG, self).__init__()

        self.model = 'gcn'  # 'gcn', 'gcn_cheby', 'dense'
        self.learning_rate = 0.02  # Initial learning rate.
        self.epochs = 20  # Number of epochs to train.
        self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.weight_decay = 0.  # Weight for L2 loss on embedding matrix.
        self.early_stopping = 10  # Tolerance for early stopping (# of epochs).
        self.max_degree = 3  # Maximum Chebyshev polynomial degree.


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.A


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features.A


class CreateTextGCNGraph():
    def __init__(self, path, dim=300):
        self.word_embeddings_dim = 300
        self.word_vector_map = {}
        self.doc_name_list = []
        self.doc_train_list = []
        self.doc_test_list = []
        self.dataset = path
        self.path = path
        self.clean_docs = self.remove_words()
        self.train_idx_orig = []
        self.test_idx_reorder = []
        self.support = []
        self.che_support = []
        self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask, self.train_size, self.test_size = self.load_corpus()
        self.t_features, self.t_y_train, self.t_y_val, self.t_y_test, self.t_train_mask, self.tm_train_mask = \
            self.post_process(self.adj, self.features,self.y_train, self.y_val, self.y_test, self.train_mask,self.val_mask, self.test_mask,self.train_size,self.test_size)

    def get_item(self):
        return self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask, self.train_size, self.test_size

    def get_titem(self):
        return self.t_features, self.t_y_train, self.t_y_val, self.t_y_test, self.t_train_mask, self.tm_train_mask

    def remove_words(self):
        doc_content_list = []
        with open(self.path, 'r') as f:
            sr = ''
            for line in f.readlines():
                if not line.startswith('>'):
                    sr += line.replace('\n', '')
                elif sr != '':
                    doc_content_list.append(sr)
                    sr = ''
            doc_content_list.append(sr)

        clean_docs = []
        for doc_content in doc_content_list:
            words = list(doc_content)
            doc_words = []
            for word in words:
                doc_words.append(word)
            doc_str = ' '.join(doc_words).strip()
            clean_docs.append(doc_str)

        return clean_docs

    def build_graph(self):
        with open(self.path, 'r') as f:
            index = 0
            for line in f.readlines():
                if line.startswith('>'):
                    line = line.split('|')
                    line[2] = line[2].replace('\n', '')
                    new_line = str(index) + '\t' + line[2] + '\t' + line[1]
                    self.doc_name_list.append(new_line.strip())
                    if line[2].find('testing') != -1:
                        self.doc_test_list.append(new_line.strip())
                    elif line[2].find('training') != -1:
                        self.doc_train_list.append(new_line.strip())
                    index += 1

        doc_content_list = []
        for line in self.clean_docs:
            doc_content_list.append(line.strip())

        train_ids = []
        for train_name in self.doc_train_list:
            train_id = self.doc_name_list.index(train_name)
            train_ids.append(train_id)
        random.shuffle(train_ids)

        # goal  训练集上的索引
        train_ids_str = [str(index) for index in train_ids]
        self.train_idx_orig = train_ids_str
        test_ids = []
        for test_name in self.doc_test_list:
            test_id = self.doc_name_list.index(test_name)
            test_ids.append(test_id)
        random.shuffle(test_ids)

        # goal  测试集的索引
        test_ids_str = [str(index) for index in test_ids]
        self.test_idx_reorder = test_ids_str
        ids = train_ids + test_ids

        # ---------------------------------
        shuffle_doc_name_list = []
        shuffle_doc_words_list = []
        for id in ids:
            shuffle_doc_name_list.append(self.doc_name_list[int(id)])
            shuffle_doc_words_list.append(doc_content_list[int(id)])

        # goal shuffle 后的文档和对应的索引
        shuffle_doc_name_str = shuffle_doc_name_list
        shuffle_doc_words_str = shuffle_doc_words_list

        word_freq = {}
        word_set = set()
        for doc_words in shuffle_doc_words_list:
            words = doc_words.split()
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        vocab = list(word_set)
        vocab_size = len(vocab)
        word_doc_list = {}
        #  构建所有的单词对应的文档id
        for i in range(len(shuffle_doc_words_list)):
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)
        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)
        word_id_map = {}
        # ATCG 对应 四个 0,1,2,3
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i
        # goal
        vocab_str = vocab

        label_set = set()
        for doc_meta in shuffle_doc_name_list:
            temp = doc_meta.split('\t')
            label_set.add(temp[2])
        label_list = list(label_set)
        # goal
        label_list_str = label_list

        # ============================================================
        train_size = len(train_ids)
        val_size = int(0.1 * train_size)
        real_train_size = train_size - val_size

        real_train_doc_names = shuffle_doc_name_list[:real_train_size]

        # goal 从训练集中选择的一部分doc id
        real_train_doc_names_str = real_train_doc_names

        row_x = []
        col_x = []
        data_x = []

        for i in range(real_train_size):
            # 每个特征300维向量
            doc_vec = np.array([0.0 for k in range(self.word_embeddings_dim)])
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embeddings_dim):
                row_x.append(i)
                col_x.append(j)
                data_x.append(doc_vec[j] / doc_len)
        # 构建了文档及嵌入矩阵
        x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
            real_train_size, self.word_embeddings_dim))

        y = []
        for i in range(real_train_size):
            doc_meta = shuffle_doc_name_list[i]
            temp = doc_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)
        y = np.array(y)

        test_size = len(test_ids)

        row_tx = []
        col_tx = []
        data_tx = []
        for i in range(test_size):
            doc_vec = np.array([0.0 for k in range(self.word_embeddings_dim)])
            doc_words = shuffle_doc_words_list[i + train_size]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embeddings_dim):
                row_tx.append(i)
                col_tx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

        tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                           shape=(test_size, self.word_embeddings_dim))

        ty = []
        for i in range(test_size):
            doc_meta = shuffle_doc_name_list[i + train_size]
            temp = doc_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            ty.append(one_hot)
        ty = np.array(ty)

        word_vectors = np.random.uniform(-0.01, 0.01,
                                         (vocab_size, self.word_embeddings_dim))

        for i in range(len(vocab)):
            word = vocab[i]
            if word in self.word_vector_map:
                vector = self.word_vector_map[word]
                word_vectors[i] = vector

        row_allx = []
        col_allx = []
        data_allx = []
        # 将文档和词拼接
        for i in range(train_size):
            doc_vec = np.array([0.0 for k in range(self.word_embeddings_dim)])
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)
            for j in range(self.word_embeddings_dim):
                row_allx.append(int(i))
                col_allx.append(j)
                data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        for i in range(vocab_size):
            for j in range(self.word_embeddings_dim):
                row_allx.append(int(i + train_size))
                col_allx.append(j)
                data_allx.append(word_vectors.item((i, j)))

        row_allx = np.array(row_allx)
        col_allx = np.array(col_allx)
        data_allx = np.array(data_allx)

        allx = sp.csr_matrix(
            (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, self.word_embeddings_dim))
        ally = []
        for i in range(train_size):
            doc_meta = shuffle_doc_name_list[i]
            temp = doc_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            ally.append(one_hot)

        for i in range(vocab_size):
            one_hot = [0 for l in range(len(label_list))]
            ally.append(one_hot)

        ally = np.array(ally)

        print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
        #  kmer
        window_size = 3
        windows = []

        for doc_words in shuffle_doc_words_list:
            words = doc_words.split()
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                # print(length, length - window_size + 1)
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)
                    # print(window)

        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        word_pair_count = {}  # 计算共现的次数
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        row = []
        col = []
        weight = []

        num_window = len(windows)

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
            if pmi <= 0:
                continue
            # 利用每个ATCG 和 文档的关系计算pmi 然后得到两个之间的特征
            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(pmi)

        doc_word_freq = {}

        for doc_id in range(len(shuffle_doc_words_list)):
            doc_words = shuffle_doc_words_list[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        for i in range(len(shuffle_doc_words_list)):
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + vocab_size)
                col.append(train_size + j)
                idf = log(1.0 * len(shuffle_doc_words_list) /
                          word_doc_freq[vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)

        node_size = train_size + vocab_size + test_size
        adj = sp.csr_matrix(
            (weight, (row, col)), shape=(node_size, node_size))

        return x, y, tx, ty, allx, ally, adj

    def load_corpus(self):
        # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
        objects = []
        for i in self.build_graph():
            objects.append(i)

        x, y, tx, ty, allx, ally, adj = tuple(objects)
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))
        train_idx_orig = self.train_idx_orig
        train_size = len(train_idx_orig)

        val_size = train_size - x.shape[0]
        test_size = tx.shape[0]

        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + val_size)
        idx_test = range(allx.shape[0], allx.shape[0] + test_size)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        features = sp.identity(features.shape[0])
        features = preprocess_features(features)
        self.support = [preprocess_adj(adj)]
        self.che_support = []
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

    def post_process(self, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size,
                     test_size):
        t_features = torch.from_numpy(features)
        t_y_train = torch.from_numpy(y_train)
        t_y_val = torch.from_numpy(y_val)
        t_y_test = torch.from_numpy(y_test)
        t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
        tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
        return t_features, t_y_train, t_y_val, t_y_test, t_train_mask, tm_train_mask

    def get_suppport(self, num):
        support = []
        num_support = 1
        if num == 0:
            support = self.support
        elif num == 1:
            support = self.che_support
            num_support = 1 + 3
        else:
            print('invalid parameter')
        t_support = []
        for i in range(len(support)):
            t_support.append(torch.Tensor(support[i]))
        return t_support, num_support