import pickle
# import pandas as pd
# import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

FILE_PATH = 'd:/work/source/kaggle/quora/'
EMBEDDING_PATH = 'd:/work/source/word_vectors/glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH = 30  # 时间步
MAX_NB_WORDS = 200000  # token时处理的最大单词数量
EMBEDDING_DIM = 100  # EMBEDDING_PATH的vocab的维度
VALIDATION_SPLIT = 0.1  # 验证集分割因子

nums_lstm = np.random.randint(175, 275)  # lstm单元数
nums_dense = np.random.randint(100, 150)  # 全连接层单元数
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
# STAMP = 'lstm_{}_{}_{:.2f}_{:.2f}'.format(nums_lstm, nums_dense, rate_drop_lstm, rate_drop_dense)

# acti = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set


def model_conv1d_(emb_matrix):

    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=30,
        trainable=False
    )

    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(30,))
    seq2 = Input(shape=(30,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2 * 32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2 * 32,))([mergea, mergeb])

    # Add the magic features
    magic_input = Input(shape=(3,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # Add the distance features (these are now TFIDF (character and word), Fuzzy matching,
    # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    # distance_input = Input(shape=(20,))
    # distance_dense = BatchNormalization()(distance_input)
    # distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer
    merge = concatenate([diff, mul, magic_dense])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2, magic_input], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


if __name__ == '__main__':
    leaks, nb_words, embedding_matrix = pickle.load(open(FILE_PATH + 'pre/leak_nb_mat', 'rb'))
    test_id, test_data1, test_data2, test_leaks = pickle.load(open(FILE_PATH + 'pre/test', 'rb'))
    train_input_1, train_input_2, train_leaks_input, train_label = pickle.load(open(FILE_PATH + 'pre/train', 'rb'))
    valid_input_1, valid_input_2, valid_leaks_input, valid_label, valid_weight = \
        pickle.load(open(FILE_PATH + 'pre/valid', 'rb'))

    class_weight = {
        0: 1.309028344,
        1: 0.472001959,
    }

    model = model_conv1d_(embedding_matrix)
    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint('cnn1d.h5', save_best_only=True, save_weights_only=True)
    # tensorboard = TensorBoard(FILE_PATH + '/logs')

    hist = model.fit([train_input_1, train_input_2, train_leaks_input], train_label,
                     validation_data=([valid_input_1, valid_input_2, valid_leaks_input], valid_label, valid_weight),
                     epochs=30, batch_size=2048, shuffle=True,
                     class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

    model.load_weights('cnn1d.h5')
    import pickle
    pickle.dump(model, open(FILE_PATH + 'ensemble/cov1d_1', 'wb'))

    # bst_val_score = min(hist.history['val_loss'])
    # wrs_val_score = max(hist.history['val_loss'])
    # print('\n', 'best valid score is {};\nworst valid score is {}'.format(bst_val_score, wrs_val_score))
    #
    # print('\n', 'Start fine-tuning')
    #
    # preds = model.predict([test_data1, test_data2, test_leaks],
    #                       batch_size=1024, verbose=1)
    # preds += model.predict([test_data2, test_data1, test_leaks],
    #                        batch_size=1024, verbose=1)
    # preds /= 2
    #
    # submission = pd.DataFrame({'test_id': test_id, 'is_duplicate': preds.ravel()})
    # submission.to_csv('%.4f_' % bst_val_score + 'cnn1d.csv', index=False)
