import numpy as np
import pandas as pd
import csv as csv
import tensorflow as tf


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

    '''Dmax = data_df['Fare'].max()
    Dmin = data_df['Fare'].min()
    Dmean = data_df['Fare'].mean()
    data_df['Fare'] = (data_df['Fare']-Dmean)/(Dmax-Dmin)'''

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    data_df = data_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Fare'], axis=1)
    return data_df

# transform data to np.array
train_df = preprocess('train.csv')
test_df = preprocess('test.csv')

train_X = np.array(train_df.drop(['Survived'],axis=1))
train_X = train_X.T
train_Y = np.matrix(train_df['Survived'])

test_X = np.array(test_df)
test_X = test_X.T
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
# build a neural network to be a classifier
X = tf.placeholder(tf.float32, shape=[8,None])
Y = tf.placeholder(tf.float32, shape=[1,None])

W1 = tf.get_variable("W1", [8,8], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [8,1], initializer = tf.zeros_initializer())

W2 = tf.get_variable("W2", [8,8], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [8,1], initializer = tf.zeros_initializer())

W3 = tf.get_variable("W3", [6,8], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())

W4 = tf.get_variable("W4", [4,6], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", [4,1], initializer = tf.zeros_initializer())

W5 = tf.get_variable("W5", [1,4], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable("b5", [1,1], initializer = tf.zeros_initializer())

Z1 = tf.add(tf.matmul(W1,X),b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2,A1),b2)
A2 = tf.nn.relu(Z2)
Z3 = tf.add(tf.matmul(W3,A2),b3)
A3 = tf.nn.relu(Z3)
Z4 = tf.add(tf.matmul(W4,A3),b4)
A4 = tf.nn.relu(Z4)
Z5 = tf.add(tf.matmul(W5,A4),b5)

hypothesis = tf.sigmoid(Z5)
#hypothesis = hypothesis > 0.5
logits = tf.transpose(hypothesis, perm=[1, 0])
labels = tf.transpose(Y, perm=[1, 0])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels = labels))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)

predict = tf.cast(hypothesis>0.5,dtype = tf.float32)
training_accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype=tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(150001):
        cost_val,_ = sess.run([cost,optimizer],feed_dict={X:train_X,Y:train_Y})
        if step %1000 == 0:
            print(step,cost_val)
    h,c,a = sess.run([hypothesis,predict,training_accuracy],feed_dict = {X:train_X,Y:train_Y})
    print('training accuracy:',a)
    test_predict = sess.run(predict,feed_dict={X:test_X})
    print(test_predict)
# output the result
with open('predict_result.csv','w') as predict:
    writer = csv.writer(predict)
    writer.writerow(['PassengerId','Survived'])
    for i in range(test_X.shape[1]):
        writer.writerow([i+892,int(test_predict[0][i])])
