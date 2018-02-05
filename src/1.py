pickle.dump((train_input_1, train_input_2, train_leaks_input, train_label), open(FILE_PATH + 'train', 'wb'))
pickle.dump((valid_input_1, valid_input_2, valid_leaks_input, valid_label, valid_weight),
            open(FILE_PATH + 'valid', 'wb'))
pickle.dump((leaks, nb_words, embedding_matrix), open(FILE_PATH + 'leak_nb_mat', 'wb'))
pickle.dump((test_df['test_id'], test_data1, test_data2, test_leaks), open(FILE_PATH + 'test', 'wb'))
