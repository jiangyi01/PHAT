import zipfile
import os
def filiter_fasta(filename, datapath, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    train_positive = []
    train_negative = []
    test_positive = []
    test_negative = []

    for index, record in enumerate(content_split):
        if index % 2 == 1:
            continue
        recordsplit = record.split('|')
        if recordsplit[-1] == 'training':
            if int(recordsplit[-2]) == 1:
                train_positive.append(content_split[index + 1])
            else:
                train_negative.append(content_split[index + 1])
        if recordsplit[-1] == 'testing':
            if int(recordsplit[-2]) == 1:
                test_positive.append(content_split[index + 1])
            else:
                test_negative.append(content_split[index + 1])
    with open(datapath + '/train_positive.txt', 'w') as f:
        for i in train_positive:
            f.write('>\n')
            f.write(i)
            f.write('\n')
    with open(datapath + '/train_negative.txt', 'w') as f:
        for i in train_negative:
            f.write('>\n')
            f.write(i)
            f.write('\n')
    with open(datapath + '/test_positive.txt', 'w') as f:
        for i in test_positive:
            f.write('>\n')
            f.write(i)
            f.write('\n')
    with open(datapath + '/test_negative.txt', 'w') as f:
        for i in test_negative:
            f.write('>\n')
            f.write(i)
            f.write('\n')
    return None

def load_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    train_dataset = []
    train_label = []
    test_dataset = []
    test_label = []
    for index, record in enumerate(content_split):
        if index % 2 == 1:
            continue
        recordsplit = record.split('|')
        if recordsplit[-1] == 'training':
            train_label.append(int(recordsplit[-2]))
            train_dataset.append(content_split[index + 1])
        if recordsplit[-1] == 'testing':
            test_label.append(int(recordsplit[-2]))
            test_dataset.append(content_split[index + 1])
    return train_dataset, train_label, test_dataset, test_label

def load_test_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    test_dataset = []

    for index, record in enumerate(content_split):
        if index % 2 == 1:
            continue
        test_dataset.append(content_split[index + 1])
    return test_dataset

def txt2fasta(trainpos, trainneg, testpos, testneg, new):
    with open(new , 'w') as f:
        with open(trainpos, 'r') as file:
            content = file.read()
            content_split = content.split('\n')
            for i in content_split:
                f.write('>pos|1|training\n')
                f.write(i)
                f.write('\n')
        with open(trainneg, 'r') as file:
            content = file.read()
            content_split = content.split('\n')
            for i in content_split:
                f.write('>neg|0|training\n')
                f.write(i)
                f.write('\n')
        with open(testpos, 'r') as file:
            content = file.read()
            content_split = content.split('\n')
            for i in content_split:
                f.write('>pos|1|testing\n')
                f.write(i)
                f.write('\n')
        with open(testneg, 'r') as file:
            content = file.read()
            content_split = content.split('\n')
            for i in content_split:
                f.write('>neg|0|testing\n')
                f.write(i)
                f.write('\n')

def txt2fasta_one(trainpos, trainneg, testpos, testneg, new):
    with open(new , 'w') as f:
        with open(trainpos, 'r') as file:
            content = file.read()
            content_split = content.split('\n')
            for i in content_split:
                f.write('>pos|1|testing\n')
                f.write(i)
                f.write('\n')


def file_name_listdir(file_dir):
    file_list=[]
    for files in os.listdir(file_dir):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        file_list.append(files)
    return file_list


if __name__ == '__main__':
    txt2fasta('../data/suplementdata/6mA_R.chinensis/train_pos.txt',
              '../data/suplementdata/6mA_R.chinensis/train_neg.txt',
              '../data/suplementdata/6mA_R.chinensis/test_pos.txt',
              '../data/suplementdata/6mA_R.chinensis/test_neg.txt',
              '../data/6mA_R.chinensis.txt')

# if __name__ == '__main__':
#     filename = '../data/testFile.txt'
#     with open(filename, 'r') as file:
#         content = file.read()
#         # print(content)
#         # > AT1G22840.1_532 | 1 | training
#         # AGATGAGGCTTTTTTACTTTGCTATATTCTTTTGCCAAATAAAATCTCAAACTTTTTTTGTTTATCATCAATTACGTTCTTGGTGGGAATTTGGCTGTAAT
#         # > AT1G44000.1_976 | 1 | training
#         # GATTCGACATAAGTCTATCTTCCATACCTTATTTACGTTTCTTCTGTGAGACAAAGTTGTACATTCTCCTGTGTTTTTTTTTGCAAATGATGTAGATTTCT
#
#     content_split = content.split('\n')
#     # print(content_split)
#     train_dataset = []
#     train_label = []
#     test_dataset = []
#     test_label = []
#     for index, record in enumerate(content_split):
#         if index % 2 == 1:
#             continue
#         recordsplit = record.split('|')
#         if recordsplit[-1] == 'training':
#             train_label.append(int(recordsplit[-2]))
#             train_dataset.append(content_split[index + 1])
#         if recordsplit[-1] == 'testing':
#             test_label.append(recordsplit[-2])
#             test_dataset.append(content_split[index + 1])
#     print(train_dataset)
#     print(train_label)
#     # import torch
#     # torch.cuda.LongTensor(train_label), torch.cuda.LongTensor(train_dataset)
#
#     # ['>AT1G22840.1_532|1|training',
#     #  'AGATGAGGCTTTTTTACTTTGCTATATTCTTTTGCCAAATAAAATCTCAAACTTTTTTTGTTTATCATCAATTACGTTCTTGGTGGGAATTTGGCTGTAAT',
#     #  '>AT1G44000.1_976|1|training',
#     #  'GATTCGACATAAGTCTATCTTCCATACCTTATTTACGTTTCTTCTGTGAGACAAAGTTGTACATTCTCCTGTGTTTTTTTTTGCAAATGATGTAGATTTCT',
#     #  '>AT1G09770.1_2698|1|training',
#     #  'ACTGGAGAGGAAGAGGACATAGCCATAGCCATGGAAGCTTCTGCATAAAAACTTGAGTTTTGTATTGCTTACAAGTTTTAAGGAGACGTAGCTTGACTTTG',
#     #  '>AT1G09645.1_586|1|training',
#     #  'ACAAAGGCCTCATGTTTGTTTGTGTTCGTTTGTCTGAGCATGTAGGTGGAACTTATCACTTATGGGTATTTAAATTTGAAGTATATATATACGCATACTTT',
#     #  '>AT1G22850.1_1097|1|training',
#     #  'ATGCTATAAAGGATATTGATGATGATGAGAAGAGAGATGCAAAGTAGGAAACAAGCCAGCGATTGGATAATGGTTTTGACTCTCTAGGATTTGTAAAACGC',
#     #  '>AT1G74960.2_2112|1|training',
