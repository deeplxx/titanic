import re
import pandas as pd
import numpy as np
import collections

# from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler


def clean_wordlist(text, remove_stopwords=False, stem_words=False):
    """将文本进行清洗处理

    Args:
        text: 输入文本 str
        remove_stopwords: 删除停用词
        stem_words: 提取词干（去除词后缀）

    Returns: 清洗后的数据 str

    """
    text = text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words('english'))
        text = [w for w in text if w not in stops]

    # clean the text
    text = ' '.join(text)

    text = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", "not ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g", " eg ", text)
    text = re.sub(r" b g", " bg ", text)
    text = re.sub(r" u s", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = ' '.join(stemmed_words)

    return text


# define some constant ########################
#
FILE_PATH = 'd:/work/source/kaggle/quora/'
EMBEDDING_PATH = 'd:/work/source/word_vectors/glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH = 30  # 时间步
MAX_NB_WORDS = 200000  # token时处理的最大单词数量
EMBEDDING_DIM = 100  # EMBEDDING_PATH的vocab的维度
VALIDATION_SPLIT = 0.1  # 验证集分割因子

nums_lstm = np.random.randint(175, 275)  # lstm隐层单元数
nums_dense = np.random.randint(100, 150)  # 全连接层单元数
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
STAMP = 'lstm_{}_{}_{:.2f}_{:.2f}'.format(nums_lstm, nums_dense, rate_drop_lstm, rate_drop_dense)

acti = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

# index word vectors ###########################
#
print('Indexing word vectors...')
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=True)
embedding_index = dict()  # 这是一个每个单词对应一个100维的向量
f = open(EMBEDDING_PATH, 'r', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    embedding_index[word] = np.asarray(values[1:], dtype='float32')
print('Found {} word vectors of word2vec\n'.format(len(embedding_index)))

# process text in dateset
#
print('Processing text dateset...\n')
train_df = pd.read_csv(FILE_PATH + 'train.csv')
test_df = pd.read_csv(FILE_PATH + 'test.csv')
train_df[['question1', 'question2']] = train_df[['question1', 'question2']].\
    astype(str).applymap(clean_wordlist)
test_df[['question1', 'question2']] = test_df[['question1', 'question2']].\
    astype(str).applymap(clean_wordlist)

train_text1 = train_df['question1'].tolist()
train_text2 = train_df['question2'].tolist()
test_text1 = test_df['question1'].tolist()
test_text2 = test_df['question2'].tolist()
# train_text = ' '.join(train_df['question1']) + ' '.join(train_df['question2'])
# test_text = ' '.join(test_df['question1']) + ' '.join(test_df['question2'])
# train_text = pd.concat([train_text1, train_text2])
# test_text = pd.concat([test_text1 + test_text2])
train_text = train_text1 + train_text2
test_text = test_text1 + test_text2
label = train_df['is_duplicate']

# Tokenizer #################################
# 把所有的单词对应到索引 -> 一个句子就是一个二维向量
#
print('Starting token...')
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)  # 处理最常见的NB个单词
tokenizer.fit_on_texts(train_text + test_text)  # 处理后每个单词对应一个索引（一个整数）
word_index = tokenizer.word_index
print('Found {} unique tokens\n'.format(len(word_index)))  # train + test全部传入

train_sequences1 = tokenizer.texts_to_sequences(train_text1)
test_sequences1 = tokenizer.texts_to_sequences(test_text1)
train_sequences2 = tokenizer.texts_to_sequences(train_text2)
test_sequences2 = tokenizer.texts_to_sequences(test_text2)

print('Starting pad seq...\n')
train_data1 = pad_sequences(train_sequences1, MAX_SEQUENCE_LENGTH)
test_data1 = pad_sequences(test_sequences1, MAX_SEQUENCE_LENGTH)
train_data2 = pad_sequences(train_sequences2, MAX_SEQUENCE_LENGTH)
test_data2 = pad_sequences(test_sequences2, MAX_SEQUENCE_LENGTH)

# generate leaky features ###################
#
ques = pd.concat([train_df[['question1', 'question2']], test_df[['question1', 'question2']]], axis=0).\
    reset_index()
q_dict = collections.defaultdict(set)


for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])


def q1_freq(row):
    """ 返回question1对应的ques个数"""
    return len(q_dict[row['question1']])


def q2_freq(row):
    """ 返回question2对应的ques个数 """
    return len(q_dict[row['question2']])


def q1_q2_intersect(row):
    """ 返回一个样本的q1和q2都对应的ques个数 """
    return len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))

train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1)

test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1)

leaks = train_df[['q1_freq', 'q2_freq', 'q1_q2_intersect']]
test_leaks = test_df[['q1_freq', 'q2_freq', 'q1_q2_intersect']]

scaler = StandardScaler()
scaler.fit(np.vstack((leaks, test_leaks)))
leaks = scaler.transform(leaks)
test_leaks = scaler.transform(test_leaks)

# embedding #################################
#
print('Preparing embedding matrix...')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

# for word, i in word_index.items():
#     if word in embedding_index:
#         embedding_matrix[i] = embedding_index[word]  # 这里进行了两次查找，可以优化！！！！！
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# np.sum 沿着轴方向，分别求轴上每一个元素的其他轴上的和
print('Null word embedding: {}\n'.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))

# sample data ###############################
#
perm = np.random.permutation(len(train_text1))  # 随机重排序列，也就是进行随机采样
mid = int(len(train_text1) * (1 - VALIDATION_SPLIT))
idx_train = perm[: mid]
idx_valid = perm[mid:]

train_input_1 = np.vstack([train_data1[idx_train], train_data2[idx_train]])
train_input_2 = np.vstack([train_data2[idx_train], train_data1[idx_train]])
train_leaks_input = np.vstack([leaks[idx_train], leaks[idx_train]])
train_label = np.concatenate([label[idx_train], label[idx_train]])

valid_input_1 = np.vstack([train_data1[idx_valid], train_data2[idx_valid]])
valid_input_2 = np.vstack([train_data2[idx_valid], train_data1[idx_valid]])
valid_leaks_input = np.vstack([leaks[idx_valid], leaks[idx_valid]])
valid_label = np.concatenate([label[idx_valid], label[idx_valid]])

valid_weight = np.ones(len(valid_label))
if re_weight:
    valid_weight *= 0.472001959
    valid_weight[valid_label == 0] = 1.309028344


# define model ###############################
#

# 经过embedding会将二维输入数据变成三维数据
embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
lstm_layer = LSTM(nums_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_input_1 = Input(shape=[MAX_SEQUENCE_LENGTH], dtype='int32')
embedding_sequence_1 = embedding_layer(sequence_input_1)
x1 = lstm_layer(embedding_sequence_1)

sequence_input_2 = Input(shape=[MAX_SEQUENCE_LENGTH], dtype='int32')
embedding_sequence_2 = embedding_layer(sequence_input_2)
x2 = lstm_layer(embedding_sequence_2)

leaks_input = Input(shape=[leaks.shape[1]])
leaks_dense = Dense(nums_dense // 2, activation=acti)(leaks_input)

merged = concatenate([x1, x2, leaks_dense])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = Dense(nums_dense, activation=acti)(merged)  # 全连接层
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

# add class weight #############################
#
if re_weight:
    class_weight = {
        0: 1.309028344,
        1: 0.472001959,
    }
else:
    class_weight = None

# train ########################################
#

model = Model([sequence_input_1, sequence_input_2, leaks_input], preds)
model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])
print('\n', STAMP)
print('\n', model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # 3轮无增长则停止
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
tensorboard = TensorBoard(FILE_PATH + '/pre/logs/lstm')

hist = model.fit([train_input_1, train_input_2, train_leaks_input], train_label,
                 validation_data=([valid_input_1, valid_input_2, valid_leaks_input], valid_label, valid_weight),
                 epochs=30, batch_size=2048, shuffle=True,
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint, tensorboard])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
wrs_val_score = max(hist.history['val_loss'])
print('\n', 'best valid score is {};\nworst valid score is {}'.format(bst_val_score, wrs_val_score))

# add class weight #############################
#
if re_weight:
    class_weight = {
        0: 1.309028344,
        1: 0.472001959,
    }
else:
    class_weight = None

# predict #######################################
#
print('\n', 'Start fine-tuning')


preds = model.predict([test_data1, test_data2, test_leaks],
                      batch_size=1024, verbose=1)
preds += model.predict([test_data2, test_data1, test_leaks],
                       batch_size=1024, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id': test_df['test_id'], 'is_duplicate': preds.ravel()})
submission.to_csv('%.4f_' % bst_val_score + STAMP + '.csv', index=False)
