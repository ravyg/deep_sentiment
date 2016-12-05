import tensorflow as tf 
import numpy as np 
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
import argparse
import csv
import warnings 
from sklearn.preprocessing import LabelEncoder
# Import data
from tensorflow.examples.tutorials.mnist import input_data

from mnist import MNIST
import matplotlib.pyplot as plt

  

def convertOneHot_data2(data):
	y = np.array([int(i) for i in data])
	#printy[:20]
	rows = len(y)
	columns = y.max() + 1
	a = np.zeros(shape=(rows,columns))
	for i,j in enumerate (y):
		a[i][j] =1
	return (a)



#######for logistic Regression############################################
def inference_logistic(x_tf,A,B):
	init=tf.constant_initializer(value=0)
	W=tf.get_variable("W",[A,B],initializer=init) ## 1,1 input and output (one layer to next)
	b=tf.get_variable("b",[B],initializer=init)
	output=tf.nn.softmax(tf.matmul(x_tf,W) + b)
	return output

#######Layers############################################
def layer(input,weight_shape,bias_shape):
	weight_stdev = (2.0/weight_shape[0])**0.5
	b_init=tf.constant_initializer(value=0)
	w_init=tf.random_normal_initializer(stddev=weight_stdev)
	W=tf.get_variable("W", weight_shape,initializer=w_init)
	b=tf.get_variable("b",bias_shape,initializer=b_init)
	return tf.nn.relu(tf.matmul(input,W) + b)

#######for DeepNet2Layers############################################
def inference_DeepNet2Layers(x_tf,A,B):
	with tf.variable_scope("hiddine_1"):
		hiddine_1=layer(x_tf,[A,10],[10])
	with tf.variable_scope("hiddine_2"):
		hiddine_2=layer(hiddine_1,[10,6],[6])	
	with tf.variable_scope("output"):
		output=layer(hiddine_2,[6,B],[B])	
	return output
#################################################
def loss_DeepNet2Layers(output,y_tf):
	xentrophy=tf.nn.softmax_cross_entropy_with_logits(output,y_tf)
	loss = tf.reduce_mean(xentrophy)
	return (loss)

###################single layer#####################################################
def inference_singlelayer(x_tf,A,B):
	with tf.variable_scope("hiddine_1"):
		output=layer(x_tf,[A,B],[B])
	return output

###########################################################################
accuracy_scores_list =[]
precision_scores_list = []
def print_stats_metric(y_test,y_pred):
	print '--------print metrices----------------'
	print ('accuracy: %.2f' % accuracy_score(y_test,y_pred))
	accuracy_scores_list.append(accuracy_score(y_test,y_pred))
	print('Precision: %.3f' % precision_score(y_test,y_pred))
	print ("Recall %.3f" % recall_score(y_true, y_pred))
	print ("f1_score %.3f" % f1_score(y_true, y_pred))
	print "confusion_matrix"
	print confusion_matrix(y_true, y_pred)
	# fpr, tpr, tresholds = roc_curve(y_true, y_pred)


###########################################################################
def plot_metric_per_epoch():
	x_epochs = []
	y_epochs =[]
	for i, val in enumerate(accuracy_scores_list):
		x_epochs.append(i)
		y_epochs.append(val)
	plt.scatter(x_epochs,y_epochs,s=50,c='lightgreen',marker='s', label='score')
	plt.xlabel('epochs')
	plt.ylabel('score')
	plt.title('score per epoch')
	plt.grid()
	plt.show()
################logiticregressioin loss function#################################
def loss(output,y_tf):
	output2= tf.clip_by_value(output,1e-10, 1.0)
	dot_product = y_tf *tf.log(output2)
	xentrophy = -tf.reduce_sum(dot_product,reduction_indices=[1])
	los =tf.reduce_mean(xentrophy)
	return (los)

#################################################
def training(cost):
	train_step=tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
	return train_step

###############################################
def evaluate(output,y_tf):
	correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(y_tf,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	return (accuracy)
#################### loading data#########################################

    
######mnist ######################
# mndata = MNIST('/home/pwrs/Documents/python-mnist/data')
# X_train,y_train= mndata.load_training()

# X_test,y_test= mndata.load_testing()
# print len(y_test)
# print len(X_test)
data_X = []
data_y = []

csvfile= list(csv.reader(open ('./feature_vector.csv', 'rU'), delimiter =','))

for data in csvfile:
   data_y.append(data[8])
   data_X.append(data[1:7])
le =LabelEncoder()
y=le.fit_transform(data_y)
X = data_X #data





######irisdata##########
# iris=datasets.load_iris()
# Xirs=iris.data[:,[1,3]]
# yirs=iris.target


# # X = mnistdata.train.images #data
# # print X
# # y= mnist.train.labels #lable of data
# X = Xirs #data
# y= yirs #lable of data
# #loading MNIST data
# X = matrix_data[:,1:] #data
# y= matrix_data[:,0] #lable of data

#########split data###########################################################

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=42)

##############normalize###################
sc= StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
########OneHotEncoding############################################

y_train_onehot=convertOneHot_data2(y_train)
y_test_onehot=convertOneHot_data2(y_test)

############################################################
A = X_train.shape[1] # number of  features
B = y_train_onehot.shape[1] # number of classes

############################################################

x_tf=tf.placeholder(tf.float32, [None,A]) # features
y_tf=tf.placeholder(tf.float32,[None,B]) # correct lables for x sammple

############################################################

#############deep nuralet 2 layer##########################
output =inference_DeepNet2Layers(x_tf, A, B)
cost = loss_DeepNet2Layers(output,y_tf)

# #############single layer##########################
# output =inference_singlelayer(x_tf, A, B)
# cost = loss_DeepNet2Layers(output,y_tf)

######for logistic regression###################
# output = inference_logistic(x_tf, A, B)
# cost=loss(output,y_tf)    ## where output is predicted results and y_tf is actual lables



train_op =training(cost)
# train_step=training(cost)
eval_op=evaluate(output,y_tf)

###############################
init=tf.initialize_all_variables()
sess= tf.Session()
sess.run(init)


####################################################
n_epochs=1000
batch_size = 1000 # we need to specify
num_samples_train_set = X_train.shape[0]
num_batches = int(num_samples_train_set/batch_size)
y_p_metric = tf.argmax(output,1)
#########Training set######################
for i in xrange(n_epochs):
	for batch_n in range(num_batches):
		sta = batch_n* batch_size
		end = sta + batch_size
		sess.run(train_op, feed_dict={x_tf:X_train[sta:end,:],y_tf:y_train_onehot[sta:end,:]})
	# feed={x_tf:X_train,y_tf:y_train_onehot}
	# sess.run(train_op,feed_dict=feed)
	print ("iteration %d" %i)
	result, y_result_metrics= sess.run([eval_op,y_p_metric],feed_dict={x_tf:X_test,y_tf:y_test_onehot})
	# accuracy_scores_list.append(result)
	y_true = np.argmax(y_test_onehot,1)
	print_stats_metric(y_true,y_result_metrics)
	print "Run {} , {}".format(i,result)
	# rr=raw_input()

plot_metric_per_epoch()
#
  
# #############Test set###########################
# for i in xrange(100,200):
# 	xs_test=np.array([[i]]) 
# 	#ys_test =np.array([[2*i]])
# 	# feed_test = {x:xs_test, y_:ys_test}
# 	feed_test={x:xs_test}
# 	results=sess.run(eval_op,feed_dict=feed_test)
# 	print "Run {}, {}".format(i,results)
# 	# rr=raw_input()
