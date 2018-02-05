import pickle
import pandas as pd
import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

FILE_PATH = 'd:/work/source/kaggle/quora/pre/'
EMBEDDING_PATH = 'd:/work/source/word_vectors/glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH = 30  # 时间步
MAX_NB_WORDS = 200000  # token时处理的最大单词数量
EMBEDDING_DIM = 100  # EMBEDDING_PATH的vocab的维度
VALIDATION_SPLIT = 0.1  # 验证集分割因子

nums_lstm = np.random.randint(175, 275)  # lstm单元数
nums_dense = np.random.randint(100, 150)  # 全连接层单元数
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
STAMP = 'lstm_{}_{}_{:.2f}_{:.2f}'.format(nums_lstm, nums_dense, rate_drop_lstm, rate_drop_dense)

acti = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set


leaks, nb_words, embedding_matrix = pickle.load(open(FILE_PATH + 'leak_nb_mat', 'rb'))
test_id, test_data1, test_data2, test_leaks = pickle.load(open(FILE_PATH + 'test', 'rb'))
train_input_1, train_input_2, train_leaks_input, train_label = pickle.load(open(FILE_PATH + 'train', 'rb'))
valid_input_1, valid_input_2, valid_leaks_input, valid_label, valid_weight = \
    pickle.load(open(FILE_PATH + 'valid', 'rb'))

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

submission = pd.DataFrame({'test_id': test_id, 'is_duplicate': preds.ravel()})
submission.to_csv('%.4f_' % bst_val_score + STAMP + '.csv', index=False)