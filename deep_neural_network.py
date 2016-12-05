#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# Some Parts of this code have be taken from Dr.Ricardo's ITS520 Class.
##########################################################
import csv
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score
from sklearn.metrics import confusion_matrix


###########################################################
def convertOneHot_data2(data):
    y=np.array([int(i) for i in data])
    #print y[:20]
    rows = len(y)
    columns = y.max()+1
    a = np.zeros(shape = (rows,columns))
    for i,j in enumerate(y):
        a[i][j]=1
    return (a)

#############################################################

# iris = datasets.load_iris()
# X = iris.data[:,[0,1,2,3]]
# y = iris.target
data_X = []
data_y = []
csvfile= list(csv.reader(open ('./feature_vector_balanced.csv', 'rU'), delimiter =','))
for data in csvfile:
   data_y.append(data[8])
   data_X.append(data[1:7])
le =LabelEncoder()
y=le.fit_transform(data_y)
X = data_X #data



###########################################################
accuracy_score_list = []
precision_score_list =[]
def print_stats_metrics(y_test, y_pred):
    # Accuracy.
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy_score_list.append(accuracy_score(y_test,y_pred))
    # Precision.
    precision_score_list.append(precision_score(y_test, y_pred))
    print('Presicion: %.2f' % precision_score(y_test, y_pred))
    # F1 Score.
    print ("f1_score %.3f" % f1_score(y_true, y_pred))
    # Confusion Matrix.
    print confusion_matrix(y_true, y_pred)

###########################################################

def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = []
    for i, val in enumerate(accuracy_score_list):
        x_epochs.append(i)
        y_epochs.append(val)
    plt.scatter(x_epochs, y_epochs, s=50, c='lightgreen', marker='s', label='score')
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.grid()
    plt.show()

###########################################################

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    # Initializing it differently. 
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

##########################################################

# Deep Neural Net - 2 hidden layers.
def inference_DeepNet2layers(x_tf, A, B):
    # First Hidder Layer 1
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 20],[20])
    # Inner Layer 2    
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [20, 16],[16])
    # Inner Layer 3
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [16, 8],[8])
    # Inner Layer 4
    with tf.variable_scope("hidden_4"):
        hidden_4 = layer(hidden_3, [8, 6],[6])
    # Final output Layer 
    with tf.variable_scope("output"):
        output = layer(hidden_4, [6, B], [B])
    return output


def inference_DeepNetNlayers(x_tf, A, B):
    # First Hidder Layer 1
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 20],[20])
    # Inner Layer 2    
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [20, 16],[16])
    # Inner Layer 3
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [16, 8],[8])
    # Inner Layer 4
    with tf.variable_scope("hidden_4"):
        hidden_4 = layer(hidden_3, [8, 6],[6])
    # Final output Layer 
    with tf.variable_scope("output"):
        output = layer(hidden_4, [6, B], [B])
    return output


###########################################################
# Loss (cost) Functions for 2 Layered Net.
def loss_DeepNet2layers(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss

###########################################################
#defines the network architecture
#simple logistic regression

def inference(x_tf, A, B):
    W = tf.Variable(tf.zeros([A,B]))
    b = tf.Variable(tf.zeros([B]))
    output = tf.nn.softmax(tf.matmul(x_tf, W) + b)
    return output
   
###########################################################

def loss(output, y_tf):
    dot_product = y_tf * tf.log(output)
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)#remove indices?
    loss = tf.reduce_mean(xentropy) #remove this line?
    return loss
    
###########################################################

def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(cost)
    return train_op

###########################################################
## add accuracy checking nodes

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

###########################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

sc= StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

y_train_onehot=convertOneHot_data2(y_train)
y_test_onehot=convertOneHot_data2(y_test)


A = X_train.shape[1]         # number of features
B = y_train_onehot.shape[1]   # number of classes


x_tf=tf.placeholder(tf.float32, [None, A])  # features
y_tf=tf.placeholder(tf.float32,[None, B])   ##correct label for x sample, this y_tf has to be onehot encoded

output = inference_DeepNet2layers(x_tf, A, B) ## for deep NN with 2 hidden layers
cost = loss_DeepNet2layers(output, y_tf)

#output = inference(x, A, B) ## for logistic regression
#cost = loss(output, y)

train_op = training(cost)
eval_op = evaluate(output, y_tf)

##################################################################
# Initialize and run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
n_epochs = 4000
batch_size = 1000
number_of_samples_train_test = X_train.shape[0]
num_batches = int(number_of_samples_train_test/batch_size)

##################################################################
y_p_metrics = tf.argmax(output,1)
##################################################################

for i in range(n_epochs):
    for batch_n in range(num_batches):
        sta = batch_n*batch_size
        end = sta + batch_size
        sess.run(train_op, feed_dict={x_tf:X_train[sta:end,:], y_tf:y_train_onehot[sta:end,:]})
    print("iteration %d " % i)
    print "***********************************************************"
    result, y_result_matrics = sess.run([eval_op, y_p_metrics], feed_dict={x_tf:X_test, y_tf:y_test_onehot})
    #accuracy_score_list.append(result)
    #print "Run {}, {}".format(i, result)

    y_true = np.argmax(y_test_onehot, 1)

    print_stats_metrics(y_true, y_result_matrics)
    print "Run {} , {}".format(i, result)
    #print("W: %f" % sess.run(W))

plot_metric_per_epoch()
##################################################################
print "<<<<<<DONE>>>>>>"
