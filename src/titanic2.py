import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB


def embarked_prepro(df):
    df['Embarked'] = df['Embarked'].fillna(df.Embarked.mode())

    dummies = pd.get_dummies(df.Embarked)
    df = df.join(dummies.drop('S', axis=1))
    df = df.drop('Embarked', axis=1)

    return df


def fare_prepro(df):
    scaler = preprocessing.StandardScaler()
    df['Fare'].fillna(df['Fare'].median(), inplace=True)  # inplace表示替换原始数组
    # df['Fare'] = df['Fare'].astype(np.int64)
    df['Fare_scaled'] = scaler.fit_transform(df.Fare.values.reshape(-1, 1))

    df = df.drop('Fare', axis=1)

    return df


def age_prepro(df):
    scaler = preprocessing.StandardScaler()
    average_age_titanic = df["Age"].mean()
    std_age_titanic = df["Age"].std()
    count_nan_age_titanic = df["Age"].isnull().sum()
    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,
                               size=count_nan_age_titanic)

    df.loc[np.isnan(df["Age"]), "Age"] = rand_1
    # df['Age'] = df['Age'].astype(np.int64)
    df['Age_scaled'] = scaler.fit_transform(df.Age.values.reshape(-1, 1))

    return df


def family_prepro(df):
    df['Family'] = df['Parch'] + df['SibSp']
    df.loc[df['Family'] > 0, 'Family'] = 1
    df.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    return df


def person_prepro(df):
    def get_person(passenger):
        age, sex = passenger
        return 'child' if age < 14 else 'old' if age > 62 else sex

    df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
    df.drop('Sex', axis=1, inplace=True)

    dummies = pd.get_dummies(df.Person)
    df = df.join(dummies.drop('male', axis=1))
    df.drop('Person', axis=1, inplace=True)

    return df


def name_prepro(df):
    name_title = df.Name.apply(lambda x: x.split(', ')[1])
    name_title = name_title.apply(lambda x: x.split('. ')[0])
    df['name_title'] = name_title
    df.drop('Name', axis=1, inplace=True)

    df.loc[df.name_title == 'Jonkheer', 'name_title'] = 'Master'
    df.loc[df.name_title.isin(['Ms', 'Mlle']), 'name_title'] = 'Miss'
    df.loc[df.name_title == 'Mme', 'name_title'] = 'Mrs'
    df.loc[df.name_title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir']), 'name_title'] = 'Sir'
    df.loc[df.name_title.isin(['Dona', 'Lady', 'the Countess']), 'name_title'] = 'Lady'

    dummies = pd.get_dummies(df.name_title)
    df = df.join(dummies.drop('Mr', axis=1))
    df.drop('name_title', axis=1, inplace=True)

    return df


def pclass_prepro(df):
    dummies = pd.get_dummies(df.Pclass).rename(columns=lambda x: 'class_' + str(x))
    df = df.join(dummies.drop('class_3', axis=1))
    df.drop('Pclass', axis=1, inplace=True)

    return df


def drop_prepro(df):
    df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    return df


def train_split(x_train, y_train):
    from sklearn.model_selection import train_test_split

    num_test = 0.2
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=num_test, random_state=23)

    return x_train, x_valid, y_train, y_valid


# noinspection PyTypeChecker,PyUnresolvedReferences
def importantest_feature(x, y):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    feature_list = x.columns
    feature_importance = clf.feature_importances_
    feature_importance = 100 * (feature_importance / feature_importance.max())
    fi = pd.DataFrame(feature_importance, index=feature_list)
    fi = fi.sort_values(0)[::-1]
    fi.plot(kind='bar')


def df_corr(df):
    correlation = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, vmax=1, linewidths=0.01, square=True, annot=True, cmap='YlGnBu', linecolor='white')
    plt.title('correlation between features')


def best_paramter_(x, y):
    #
    # x_train, x_test, y_train, y_test = train_split(x, y)

    # 选择一些参数进行尝试
    # clf = RandomForestClassifier()
    clf = GradientBoostingClassifier()

    paramters = {
        'n_estimators': [30000],  # 80
        # 'max_features': ['log2', 'sqrt'],  # sqrt
        # 'criterion': ['entropy', 'gini'],  # gini
        'max_depth': [5, 10],  # 20
        'min_samples_split': [2, 3],  # 3
        'min_samples_leaf': [5, 8]  # 8
    }
    acc_scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(clf, paramters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(x, y)  # 这个耗费时间
    clf = grid_obj.best_estimator_
    # clf.fit(x_train, y_train)
    #
    # predic = clf.predict(x_test)
    # print('the accuracy_score is {0}'.format(accuracy_score(y_test, predic)))
    print('the best paramters is {0}'.format(clf.get_params()))
    return clf


def run_kflod(clf, x_all, y_all):
    """ K折验证

    Args:
        clf: 分类器
        x_all: 训练集特征数据
        y_all: 训练集标签数据

    Returns: None

    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

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


def df_prepro(df):

    df = drop_prepro(df)
    df = embarked_prepro(df)
    df = fare_prepro(df)
    df = age_prepro(df)
    df = family_prepro(df)
    df = person_prepro(df)
    df = name_prepro(df)
    df = pclass_prepro(df)
    df = df.drop(['Master', 'child', 'Dr', 'Sir', 'old', 'Rev', 'Lady', 'Age'], axis=1)

    return df


if __name__ == '__main__':
    file_path = 'd:/work/source/kaggle/titanic/'
    train_df = pd.read_csv(file_path + 'train.csv')
    test_df = pd.read_csv(file_path + 'test.csv')
    # df_corr(train_df)

    x_df = train_df.drop('Survived', axis=1)
    y_df = train_df['Survived']

    df = df_prepro(x_df)
    # print(df.sample(3))
    # importantest_feature(df, y_df)

    x_train, x_valid, y_train, y_valid = train_split(df, y_df)
    # clf = best_paramter_(x_train, y_train)
    # clf = RandomForestClassifier()
    # clf = GradientBoostingClassifier(max_depth=5, n_estimators=30000)
    # clf.fit(x_train, y_train)
    import xgboost as xgb
    clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05)
    clf.fit(x_train, y_train)
    # run_kflod(clf, x_train, y_train)
    valid_pred = clf.predict(x_valid)
    accuracy_ = accuracy_score(y_valid, valid_pred)
    print('valid accuracy: {0}'.format(accuracy_))

    test_id = test_df['PassengerId']
    test_df_ = df_prepro(test_df)
    clf.fit(df, y_df)
    test_pred = clf.predict(test_df_)

    out = pd.DataFrame({'PassengerId': test_id, 'Survived': test_pred})
    out.to_csv(file_path + 'pred3.csv', index=False)
