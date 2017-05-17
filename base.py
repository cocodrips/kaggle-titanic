import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns

from sklearn.cross_validation import train_test_split
import seaborn as sns

exp_values = [
    'Pclass',
    #     'PassengerId',
    'C(Sex)',
    'C(Child)',
    'C(Embarked)',
    #     'SibSp',
]
predict_value = 'Survived'
child_threthold = 16

survived_ticket_set = {
    '110152',
    '113760',
    '13502',
    '1601',
    '24160',
    '2666',
    '29106',
    '347077',
    '347742',
    'PC 17572',
    'PC 17755',
    'PC 17757'
}

dead_ticket_set = {
    '3101295',
    '345773',
    '347082',
    '347088',
    '349909',
    '382652',
    'CA 2144',
    'LINE',
    'S.O.C. 14879',
    'W./C. 6608'
}


def output_result(model, category_data, base_data):
    result = model.predict(category_data)
    with open('data/result.csv', 'w') as f:
        f.write('PassengerId,Survived\n')
        for r, pid in zip(result, base_data['PassengerId']):
            f.write('{},{}\n'.format(pid, str(int(r))))


def category_to_table(titanic, is_training_data=True):
    fomula = '+'.join(exp_values)
    if is_training_data:
        fomula = fomula + '+' + predict_value

    print('fomula:', fomula)

    c_df = patsy.dmatrix(fomula, data=titanic, return_type='dataframe')
    return c_df


def load_titanic_data(is_training_data=True):
    """
    データを読みこみ、
    必要なパラメータに絞る
    """

    s_rate_ticket_dict = dict()
    s_count_ticket_dict = dict()

    def is_child(age):
        if pd.isnull(age):
            return False
        if age < child_threthold:
            return True
        return False

    def ticket_count_info(x):
        if x in s_count_ticket_dict:
            return s_count_ticket_dict[x]
        return 0

    def ticket_ratio_info(x):
        if x in s_rate_ticket_dict:
            return s_rate_ticket_dict[x]
        return None

    titanic = None
    if is_training_data:
        titanic = pd.read_csv("data/train.csv")
    else:
        titanic = pd.read_csv("data/test.csv")

    if is_training_data:
        subset_ = ['Age', 'Pclass']
        titanic = titanic.dropna(
            subset=subset_
        )

        for k, v in titanic.groupby('Ticket')[['Survived']].count().items():
            print(k)
            for kk, vv in v.items():
                s_count_ticket_dict[kk] = vv

        for k, v in titanic.groupby('Ticket')[['Survived']].mean().items():
            print(k)
            for kk, vv in v.items():
                s_rate_ticket_dict[kk] = vv

        titanic = titanic.assign(
            TicketCount=titanic['Ticket'].map(ticket_count_info),
            TicketSRate=titanic['Ticket'].map(ticket_ratio_info),
            Type="Train",
            Training=1,
            Test=0
        )
    else:
        titanic = titanic.assign(
            Surrvived=None,
            Type="Test",
            Training=0,
            Test=1
        )

    titanic = titanic.assign(
        Child=titanic['Age'].map(is_child),
    )
    return titanic


def hit_rate(predict, test):
    N = len(predict)
    hit = sum([1 if p == t else 0 for p, t in zip(predict, test)])
    return "{0:.3f}%".format(100 * hit / N * 1.0)


def main():
    titanic = load_titanic_data()
    titanic_category = category_to_table(titanic)

    x_columns = list(titanic_category.columns)
    x_columns.remove(predict_value)

    X_train, X_test, Y_train, Y_test = train_test_split(titanic[x_columns],
                                                        titanic[predict_value],
                                                        test_size=0.4,
                                                        random_state=0)


titanic = load_titanic_data()
titanic.head(2)

## カテゴリカルなデータに変換
titanic_category = category_to_table(titanic)
titanic_category.head(2)

test_data = load_titanic_data(False)
test_data_category = category_to_table(test_data, False)

full_data = pd.concat([titanic, test_data])
