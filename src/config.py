"""config"""

train_root = ''
val_root = ''
save_root = ''

batch_size = 32    # 一次训练所选取的样本数
lr = 1e-3             # 学习率
n_epoch = 250          # 训练次数
dropout = 0.5   
d_model = 1024   # 词向量维度
n_class = 2    # 输出
vocab_size = 21   # 词典大小
nlayers = 4   # transformer encoder layer
nhead = 4    # transformer encoder head
dim_feedforward  =1024 # transformer encoder feedforward
kmers = 5 # kmer