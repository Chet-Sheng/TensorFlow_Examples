# ============================
# Tutorial - Tensorflow
# (CNN classification)
# -----------------------------
# Copyright @ dianaborsa
# =============================

#!/usr/bin/env python

import numpy as np 
import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
import random

BATCH_SIZE    = 64
MAX_ITERS     = 100 
LEARNING_RATE = 0.001


# --------------------------------------------------------------------
# Helper fct: Builds 2D convolutional layer, followed by
# (Optionally) max_pooling layer
# (Optionally) dropout layer
def get2DConvolutionalLayer(X, shape_conv, max_pooling=True, dropout=False, dropout_prob=0.5):

	w = tf.Variable(tf.random_normal(shape_conv, stddev=0.01))
	conv_layer = tf.nn.relu(tf.nn.conv2d(X,w, strides= [1,1,1,1], padding='SAME'))

	if max_pooling:
		outlayer = tf.nn.max_pool(conv_layer, ksize=[1,2,2,1], 
									strides=[1,2,2,1], padding='SAME')
	else:
		outlayer = conv_layer

	if dropout:
		outlayer = tf.nn.dropout(outlayer, dropout_prob)

	return outlayer

# --------------------------------------------------------------------
# Helper fct: Build linear layer + w/ or w/o bias compoment
def getLinearLayer(X, dimIN, dimOUT, bias=True):

	if bias:
		w = tf.Variable(tf.random_normal([dimIN, dimOUT], stddev=0.01))
		b = tf.Variable(tf.random_normal([1, dimOUT], stddev=0.01))
		return tf.matmul(X, w)+b
	else:
		w = tf.Variable(tf.random_normal([dimIN, dimOUT], stddev=0.01))
		return tf.matmul(X, w)


# --------------------------------------------------------------------
# Helper fct: Build ReLU hidden layer + optionally droupout AFTER non-linearity
def getHiddenLayer(X, dimIN, dimOUT, dropout=False, dropout_prob=0.5):
	
	# ReLU hidden layer	
	w_h = tf.Variable(tf.random_normal([dimIN, dimOUT], stddev=0.01))
	b_h = tf.Variable(tf.random_normal([1, dimOUT], stddev=0.01))
	h = tf.nn.relu(tf.matmul(X, w_h)+b_h)

	# Dropout layer
	if dropout:
		output = tf.nn.dropout(h, dropout_prob)
	else:
		output = h

	return output


# --------------------------------------------------------------------
# Build model
def getModel(X, shapeIN, shapeOUT):

	print shapeIN
	conv_layer1    = get2DConvolutionalLayer(X,           [3,3,1,32],   dropout=True, dropout_prob=0.7)
	conv_layer2    = get2DConvolutionalLayer(conv_layer1, [3,3,32,64],  dropout=True, dropout_prob=0.8)
	#conv_layer3    = get2DConvolutionalLayer(conv_layer2, [3,3,64,128], dropout=True, dropout_prob=0.8)
	
	# need to flatten output of the from (?,4,4,128) to vector of 1D vector 4*4*128
	input_hidden   = tf.reshape(conv_layer2, [-1,64*7*7])
	
	# add a fully-connected hidden layer
	hidden_layer   = getHiddenLayer(input_hidden, 64*7*7, 500)
	
	#output layer
	scoreY = getLinearLayer(hidden_layer, 500, shapeOUT)

	return scoreY


# --------------------------------------------------------------------
# Generic training function:
# --------------------------------------------------------------------
# X        		-> tf.placeholder for INPUT
# Y        		-> tf.placeholder for OUTPUT
# train_op 		-> training operation (tf.Optimizer)
# accuracy 		-> sometimes the same as the loss fct, but doesn't affect 
# 			  	   the training, just displayed evaluation
# trainX,trainY -> training data and labels
# testX, testY  -> testing data and labels
def trainModel(accuracy, train_op, X, Y, trainX, trainY, testX, testY):
	# Launch session for training
	with tf.Session() as sess:

		# initialize variables
		tf.initialize_all_variables().run()

		print "==============================="
		print "Training started........"

		for indexIter in range(MAX_ITERS):
			for startIndex, endIndex in zip( range(0,len(trainX),BATCH_SIZE), range(BATCH_SIZE,len(trainX),BATCH_SIZE)):
				sess.run(train_op, feed_dict={X: trainX[startIndex:endIndex], Y: trainY[startIndex:endIndex]})

			# Visualize accuracy only every 10 iterations
			if indexIter%1==0 :

				acc_train = 0.0
				n_batches_train=0
				for startIndex_acc, endIndex_acc in zip( range(0,len(trainX), BATCH_SIZE), range(BATCH_SIZE,len(trainX), BATCH_SIZE)):
					acc_train = acc_train + sess.run(accuracy, feed_dict={X:trainX[startIndex_acc:endIndex_acc], Y: trainY[startIndex_acc:endIndex_acc]})
					#print acc_train/n_batches_train
					n_batches_train = n_batches_train+1
				acc_train = acc_train/n_batches_train

				acc_test = 0.0
				n_batches_test = 0
				for startIndex_acc, endIndex_acc in zip( range(0,len(testX), BATCH_SIZE), range(BATCH_SIZE,len(testX), BATCH_SIZE)):
					acc_test = acc_test + sess.run(accuracy, feed_dict={X:testX[startIndex_acc:endIndex_acc], Y: testY[startIndex_acc:endIndex_acc]})
					n_batches_test = n_batches_test+1
				acc_test = acc_test/n_batches_test


				print('Iteration %d: Accuracy %.5f(training) %.5f(testing)' %(indexIter, acc_train, acc_test))
				
			# for startIndex, endIndex in zip( range(0,len(trainX),BATCH_SIZE), range(BATCH_SIZE,len(trainX),BATCH_SIZE)):
			# 	sess.run(train_op, feed_dict={X: trainX[startIndex:endIndex], Y: trainY[startIndex:endIndex]})

		print "Training finished."
		print "==============================="


def getRandomSample(X,Y, sampleSize):

	totalSamples = len(X)
	print np.array(range(totalSamples))
	suffled_indices = np.random.permutation(range(totalSamples))

	print suffled_indices
	selected_indices = suffled_indices[:sampleSize]
	return (X[selected_indices,:], Y[selected_indices,:])

def main():

	# ==================================
	# 0. Load dataset
	# ==================================
	print "Loading the data......"
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

	# (Optimal) Subsample data, as for cnn the whole set might be too large
	#(trainX, trainY) = getRandomSample(trainX, trainY, 5000)
	#(testX, testY) = getRandomSample(testX, testY, 1000)

	(nrTrainSamples, dimX) = trainX.shape
	(nrTestSamples, dimY)  = testY.shape

	IMAGE_DIM = int(dimX**0.5)

	trainX = np.reshape(trainX, (nrTrainSamples, IMAGE_DIM, IMAGE_DIM, 1))
	testX  = np.reshape(testX, (nrTestSamples, IMAGE_DIM, IMAGE_DIM, 1))

	print "Finished: data loaded. Stats below: "
	print "Nr of training samples: %d" %nrTrainSamples
	print "Nr of testing  samples: %d" %nrTestSamples
	print "Shape of X: " + str(trainX.shape)
	print "Dimension of Y: %d" %dimY
	IMAGE_DIM = int(dimX**0.5)  
	#plt.imshow(np.reshape(trainX[0,:],(IMAGE_DIM,IMAGE_DIM)))


	# =================================
	# 1. Build the model
	# =================================
	X = tf.placeholder("float", [None, IMAGE_DIM, IMAGE_DIM,1])
	Y = tf.placeholder("float", [None, dimY])

	# model
	model = getModel(X, dimX, dimY)

	# Get predicted classes
	predict_op = tf.argmax(model, dimension=1)

	# Loss function (Cross-entropy) used for training
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

	# Accuracy: proportion of right answers producted by the classifire
	accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(Y,1),predict_op), tf.float32))


	# ==================================
	# Traing model
	# ==================================
	# Gradient Descent, for some reason converges very rapidly to quite bad solution ~ 10% accuracy
	#train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss) 
	
	# Adam + RMS: 95-97% accuracy
	train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss) 
	#train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9).minimize(loss)
	trainModel(accuracy, train_op, X, Y, trainX, trainY, testX, testY)



if __name__ == '__main__':

	# Set random seed
	tf.set_random_seed(12345)

	main()