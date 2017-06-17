from IPython.core.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns

from sklearn.model_selection import train_test_split

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

    c_df = patsy.dmatrix(fomula, data=titanic, return_type='dataframe')
    return c_df


def load_titanic_data(is_training_data=True):
    """
    データを読みこみ必要なパラメータに絞る
    
    :type is_training_data:bool True: train.csv False: test.csv 
    """

    def is_child(age):
        if pd.isnull(age):
            return False
        if age < child_threthold:
            return True
        return False

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

        titanic = titanic.assign(
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


titanic: pd.DataFrame = load_titanic_data() 
titanic_female: pd.DataFrame = titanic.query('Sex=="female"')
titanic_male: pd.DataFrame = titanic.query('Sex=="male"')

titanic_category = category_to_table(titanic)
# titanic_category_female = titanic_category.query('"C(Sex)[T.male]"==0')
# titanic_category_male = titanic_category.query('"C(Sex)[T.male]"==1')

test_titanic = load_titanic_data(False)
test_titanic_category = category_to_table(test_titanic, False)

full_data = pd.concat([titanic, test_titanic])

x_columns = list(titanic_category.columns)
x_columns.remove(predict_value)

X_train, X_test, Y_train, Y_test = train_test_split(titanic_category[x_columns],
                                                    titanic_category[predict_value],
                                                    test_size=0.4,
                                                    random_state=0)



def show_basic_data():
    display('titanic_male')
    display(titanic_male.head(1))
    
    display('titanic')
    display(titanic.head(1))

    display('titanic_category')
    display(titanic_category.head(1))
