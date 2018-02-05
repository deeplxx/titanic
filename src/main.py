import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib as plt

# sns.pointplot('Sex', 'Survived', 'Pclass', data_train, palette={1: 'blue', 2: 'red', 3: 'green'})


def simplify_ages(df):
    """ 将Age分块 """
    df.Age = df.Age.fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 25, 35, 60, 100]
    group_names = ['unknown', 'baby', 'child', 'Teenager', 'student', 'young', 'adult', 'senior']
    categeries = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categeries
    return df


def simplify_cabins(df):
    """ 将Cabin取第一个字母，N代表无值 """
    df.Cabin = df.Cabin.fillna('N')

    # 取Cabin的第一个字母
    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df


def simplify_fares(df):
    """ 对fare票价进行分块 """
    df.Fare = df.Fare.fillna(-0.5)
    bins = [-1, 0, 8, 15, 32, 600]
    group_names = ['unknown', '1_qua', '2_qua', '3_qua', '4_qua']
    categeries = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categeries
    return df


def format_names(df):
    """ 对name进行划分为两列 """
    df['L_name'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['Prefix_name'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df


def drop_features(df):
    """ 丢弃无用特征 """
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)


def transform_features(df):
    """ 格式化特征 """
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_names(df)
    df = drop_features(df)
    return df


# noinspection PyUnresolvedReferences
def encode_features(df_train, df_test):
    """ 将离散特征编码 """
    from sklearn import preprocessing

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'L_name', 'Prefix_name']
    df_combined = pd.concat([df_train, df_test])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test


def run_kflod(clf, x_all, y_all):
    """ K折验证

    Args:
        clf: 分类器
        x_all: 训练集特征数据
        y_all: 训练集标签数据

    Returns: None

    """
    from sklearn.model_selection import KFold

    kf = KFold(10)
    outcomes = list()
    fold = 0

    for train_index, test_index in kf.split(x_all):
        fold += 1
        x_train, x_test = x_all.values[train_index], x_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(x_train, y_train)
        predic = clf.predict(x_test)
        accuracy = accuracy_score(y_test, predic)
        outcomes.append(accuracy)
        print('Fold {0} accuracy: {1}'.format(fold, accuracy))

    mean_outcomes = np.mean(outcomes)
    print('Mean accuracy: {0}'.format(mean_outcomes))


def preprocessing():
    # 数据预处理 ########################################################################################################
    #
    file_path = 'd:/work/source/kaggle/titanic/'
    data_train = pd.read_csv(file_path + 'train.csv')
    data_test = pd.read_csv(file_path + 'test.csv')

    data_train = transform_features(data_train)
    data_test = transform_features(data_test)

    data_train, data_test = encode_features(data_train, data_test)

    # 将训练数据分成train+valid #########################################################################################
    #
    from sklearn.model_selection import train_test_split

    x_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']

    num_test = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=23)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    # 数据预处理 ########################################################################################################
    #
    file_path = 'd:/work/source/kaggle/titanic/'
    data_train = pd.read_csv(file_path + 'train.csv')
    data_test = pd.read_csv(file_path + 'test.csv')

    data_train = transform_features(data_train)
    data_test = transform_features(data_test)

    data_train, data_test = encode_features(data_train, data_test)

    # 将训练数据分成train+valid #########################################################################################
    #
    from sklearn.model_selection import train_test_split

    x_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']

    num_test = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=23)

    # 训练 #############################################################################################################
    #
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer, accuracy_score
    from sklearn.model_selection import GridSearchCV

    clf = RandomForestClassifier()

    # 选择一些参数进行尝试
    paramters = {
        'n_estimators': [4, 6, 9],
        'max_features': ['log2', 'sqrt'],
        'criterion': ['entropy', 'gini'],
        'max_depth': [2, 3, 5, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 5, 8]
    }
    acc_scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(clf, paramters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(x_train, y_train)
    clf = grid_obj.best_estimator_
    clf.fit(x_train, y_train)

    predic = clf.predict(x_test)
    print(accuracy_score(y_test, predic))

    # k折验证 ##########################################################################################################
    #
    run_kflod(clf, x_all, y_all)

    # 预测测试数据 ######################################################################################################
    #
    ids = data_test['PassengerId']
    predic = clf.predict(data_test.drop(['PassengerId'], axis=1))

    output = pd.DataFrame({'PassengerId': ids, 'Survived': predic})
