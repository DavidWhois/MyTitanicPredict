import numpy as np
import pandas as pd
import csv as csv
from sklearn.tree import DecisionTreeClassifier


def preprocess(filename):
    data_df = pd.read_csv(filename, header=0)
    data_df['Gender'] = data_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    # All missing Embarked -> just make them embark from most common place
    if len(data_df.Embarked[ data_df.Embarked.isnull() ]) > 0:
        data_df.Embarked[ data_df.Embarked.isnull() ] = data_df.Embarked.dropna().mode().values

    Ports = list(enumerate(np.unique(data_df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    data_df.Embarked = data_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
    dummy_embarked = pd.get_dummies(data_df['Embarked'],prefix='Embarked',drop_first=False)
    data_df = pd.concat([data_df,dummy_embarked], axis = 1)
    data_df = data_df.drop(['Embarked'],axis=1)
    # All the ages with no data -> make the median of all Ages
    median_age = data_df['Age'].dropna().median()
    if len(data_df.Age[ data_df.Age.isnull() ]) > 0:
        data_df.loc[ (data_df.Age.isnull()), 'Age'] = median_age
    data_df['Age'] *= 0.1

    Dmax = data_df['Fare'].max()
    Dmin = data_df['Fare'].min()
    Dmean = data_df['Fare'].mean()
    data_df['Fare'] = (data_df['Fare']-Dmean)/(Dmax-Dmin)

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    data_df = data_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Fare','Parch','SibSp'], axis=1)
    return data_df

# transform data to np.array
train_df = preprocess('train.csv')
test_df = preprocess('test.csv')
train_X = np.array(train_df.drop(['Survived'],axis=1),dtype=np.float32)
train_Y = np.matrix(train_df['Survived']).T

test_X = np.array(test_df,dtype=np.float32)
test_X[np.isnan(test_X)] = np.mean(test_X[~np.isnan(test_X)])
#print('是否存在nan？',np.isnan(test_X).any())
print('train_X shape:',train_X.shape)
print('train_y shape:',train_Y.shape)
print('text_X shape:',test_X.shape)

clf = DecisionTreeClassifier(min_samples_split=20)
print('fitting model...')
clf.fit(train_X,train_Y)

test_predict = clf.predict(test_X)
print('finish!')
print('score in training set:',clf.score(train_X,train_Y))
# output the result
with open('Tree_predict_result.csv','w') as predict:
    writer = csv.writer(predict)
    writer.writerow(['PassengerId','Survived'])
    for i in range(test_X.shape[0]):
        writer.writerow([i+892,int(test_predict[i])])
