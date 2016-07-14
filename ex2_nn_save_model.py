# =======================================
# Tutorial - Tensorflow
# (Saving and loading models)
# -----------------------------
# Copyright @ dianaborsa
# ----------------------------------------
# Based on ex2_nn (NN classification)
# ========================================

#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import input_data
import matplotlib.pyplot as plt

BATCH_SIZE     = 32
MAX_ITERS      = 10
LEARNING_RATE  = 0.01
MODEL_FILENAME = "nn_model.ckpt" 

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
def getModel(X, dimIN, dimOUT):

	# input layer: apply dropout
	#prob_keep_input = 0.5
	#X = tf.nn.dropout(X, prob_keep_input)

	dimH1, dimH2 = 500, 500
	# hidden layer1
	h1 = getHiddenLayer(X, dimIN, dimH1, dropout=False, dropout_prob=0.5)
	# hidden layer2
	h2 = getHiddenLayer(h1, dimH1, dimH2, dropout=False, dropout_prob=0.7)

	# output layer
	scoreY = getLinearLayer(h2, dimH2, dimOUT, bias=True)

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
def trainModel(sess, accuracy, train_op, X, Y, trainX, trainY, testX, testY):
	# Launch session for training
	#with tf.Session() as sess:

		print "==============================="
		print "Training started........"

		for indexIter in range(MAX_ITERS):
			for startIndex, endIndex in zip( range(0,len(trainX),BATCH_SIZE), range(BATCH_SIZE,len(trainX),BATCH_SIZE)):
				sess.run(train_op, feed_dict={X: trainX[startIndex:endIndex], Y: trainY[startIndex:endIndex]})

			# Visualize accuracy only every 10 iterations
			if indexIter%1==0 :
				
				acc_train = sess.run(accuracy, feed_dict={X:trainX, Y:trainY})
				acc_test  = sess.run(accuracy, feed_dict={X:testX,  Y:testY})

				print('Iteration %d: Accuracy %.5f(training) %.5f(testing)' %(indexIter, acc_train, acc_test))
				
			# for startIndex, endIndex in zip( range(0,len(trainX),BATCH_SIZE), range(BATCH_SIZE,len(trainX),BATCH_SIZE)):
			# 	sess.run(train_op, feed_dict={X: trainX[startIndex:endIndex], Y: trainY[startIndex:endIndex]})

		print "Training finished."
		print "===============================\n"


def saveModel(sess, MODEL_FILENAME):

	print('Saving model at: '+MODEL_FILENAME)
	saver = tf.train.Saver()
	saver.save(sess, MODEL_FILENAME)
	print('Model succesfully saved.\n')


def loadModel(sess, MODEL_FILENAME):

	print('Loading save model from: '+MODEL_FILENAME)
	saver = tf.train.Saver()
	saver.restore(sess, MODEL_FILENAME)
	print('Model succesfully loaded.\n')


def main():

	# ==================================
	# 0. Load dataset
	# ==================================
	print "Loading the data......"
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	(nrTrainSamples, dimX) = trainX.shape
	(nrTestSamples, dimY)  = testY.shape

	print "Finished: data loaded. Stats below: "
	print "Nr of training samples: %d" %nrTrainSamples
	print "Nr of testing  samples: %d" %nrTestSamples
	print "Dimension of X: %d" %dimX
	print "Dimension of Y: %d" %dimY
	IMAGE_DIM = int(dimX**0.5)  
	#plt.imshow(np.reshape(trainX[0,:],(IMAGE_DIM,IMAGE_DIM)))


	# ==================================
	# 1. Build model
	# ==================================
	X = tf.placeholder("float",[None, dimX])
	Y = tf.placeholder("float",[None, dimY])

	# model returns a score vector over all classes
	model = getModel(X, dimX, dimY)

	
	# Get predicted classes
	predict_op = tf.argmax(model, 1)
	
	# Loss function (Cross-entropy for classification)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))
	
	# Accuracy(if different measure from loss)
	agreement = tf.equal(predict_op, tf.argmax(Y,1))
	accuracy  = tf.reduce_mean( tf.cast(agreement, tf.float32 ))


	# ==================================
	# 2. Train & Save model
	# ==================================
	train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9).minimize(loss)
	#train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss) 
	
	with tf.Session() as sess:
		# initialize variables
		tf.initialize_all_variables().run()

		# train model
		trainModel(sess, accuracy, train_op, X, Y, trainX, trainY, testX, testY)

		# check the update of sess variables
		print('CHECK: Accuracy of saved model')
		acc_train = sess.run(accuracy, feed_dict={X:trainX, Y:trainY})
		acc_test  = sess.run(accuracy, feed_dict={X:testX,  Y:testY})
		print('Accuracy: %.5f(training) %.5f(testing)' %(acc_train, acc_test))

		# save model
		saveModel(sess, MODEL_FILENAME)



	# ==================================
	# 3. Load saved model
	# ==================================
	with tf.Session() as sess:

		# initialize variables
		tf.initialize_all_variables().run()

		# load model
		loadModel(sess, MODEL_FILENAME)

		# check model
		print('CHECK: After loading save model')
		acc_train = sess.run(accuracy, feed_dict={X:trainX, Y:trainY})
		acc_test  = sess.run(accuracy, feed_dict={X:testX,  Y:testY})
		print('Accuracy: %.5f(training) %.5f(testing)' %(acc_train, acc_test))


if __name__ == '__main__':
	
	# (REPRODUCIBILITY) set random seeds
	tf.set_random_seed(123)
	
	main()