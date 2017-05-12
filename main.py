import pandas as pd
import patsy
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
titanic = pd.read_csv("data/train.csv")
titanic_nan = titanic.dropna(subset=['Age', 'Sex', 'Pclass'])

_f = 'C(Pclass) + C(Sex) + Age + Survived'
c_df = patsy.dmatrix(_f, data=titanic_nan, return_type='dataframe')


parameters = ["C(Pclass)[T.2]","C(Pclass)[T.3]","C(Sex)[T.male]","Age"]
X = c_df[parameters]
Y = c_df['Survived']

model = tree.DecisionTreeClassifier(max_depth=6)
model.fit(X, Y)
# model

model.predict(X)
test_data = pd.read_csv("data/test.csv")
_f = 'C(Pclass) + C(Sex) + Age'
test_data_cf = patsy.dmatrix(_f, data=test_data, return_type='dataframe')

for d in model.fit(test_data_cf):
    print (d)

#
#
# with open("data/result.csv", 'w') as f:
#     f.write("PassengerId,Survived\n")
#     for i in range(N):
#         d = p.iloc[i]
#         if d['Sex'].startswith('f'):
#             f.write("{},{}\n".format(d['PassengerId'], 1))
#         else:
#             f.write("{},{}\n".format(d['PassengerId'], 0))
#
#
#

