# ============================
# Tutorial - Tensorflow
# (Logistics Regression)
# -----------------------------
# Copyright @ dianaborsa
# =============================

#!/usr/bin/env python

import tensorflow as tf 
import numpy as np
import input_data
import matplotlib.pyplot as plt
import random

BATCH_SIZE    = 32
MAX_ITERS     = 100 
LEARNING_RATE = 0.01


def getModel(X, dimIN, dimOUT):
	w = tf.Variable(tf.random_normal((dimIN,dimOUT), stddev = 0.01))
	b = tf.Variable(tf.random_normal((1,dimOUT) , stddev = 0.01))
	probY = tf.matmul(X,w)+b
	return probY


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
			if indexIter%10==0 :
				# predictedY = sess.run(predict_op, feed_dict={X:testX})
				# print predictedY
				# print np.argmax(testY, axis=1)
				acc_train = sess.run(accuracy, feed_dict={X:trainX, Y:trainY})
				acc_test  = sess.run(accuracy, feed_dict={X:testX,  Y:testY})
				# print acc
				print('Iteration %d: Accuracy %.5f(training) %.5f(testing)' %(indexIter, acc_train, acc_test))
				#print(i, np.mean(np.argmax(testY, axis=1) == sess.run(predict_op, feed_dict={X: testX})))

			
		print "Training finished."
		print "==============================="



def main():

	print "Loading the data......"
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	(nrTrainSamples, dimX) = trainX.shape
	(nrTestSamples, dimY)  = testY.shape

	print "Finished: data loaded. Stats below: "
	print "Nr of training samples: %d" %nrTrainSamples
	print "Nr of testing  samples: %d" %nrTestSamples
	print "Dimension of X: %d" %dimX
	print "Dimension of X: %d" %dimY
	IMAGE_DIM = int(dimX**0.5)  
	plt.imshow(np.reshape(trainX[0,:],(IMAGE_DIM,IMAGE_DIM)))


	# Build Model
	X = tf.placeholder("float", [None, dimX])
	Y = tf.placeholder("float", [None, dimY])
	model = getModel(X, dimX, dimY)

	# Loss function
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

	# Optimizer
	train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss) 

	# Prediction/Output
	predict_op = tf.argmax(model, 1)
	# note: tf needs explicted casting after equal (don't need it in numpy): bool->float
	accuracy   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), predict_op), tf.float32))

	# Launch session for training
	with tf.Session() as sess:

		# initialize variables
		tf.initialize_all_variables().run()
		#accuracy   = tf.reduce_mean(tf.argmax(Y, dimension=1) == predict_op)

		print "==============================="
		print "Training started........"

		for indexIter in range(MAX_ITERS):
		# for startIndex, endIndex in zip( range(0,len(trainX),BATCH_SIZE), range(BATCH_SIZE,len(trainX),BATCH_SIZE)):
		# 	sess.run(train_op, feed_dict={X: trainX[startIndex:endIndex], Y: trainY[startIndex:endIndex]})

			if indexIter%10==0 :
				# predictedY = sess.run(predict_op, feed_dict={X:testX})
				# print predictedY
				# print np.argmax(testY, axis=1)
				acc_train = sess.run(accuracy, feed_dict={X:trainX, Y:trainY})
				acc_test  = sess.run(accuracy, feed_dict={X:testX,  Y:testY})
				# print acc
				print('Iteration %d: Accuracy %.5f(training) %.5f(testing)' %(indexIter, acc_train, acc_test))
				#print(i, np.mean(np.argmax(testY, axis=1) == sess.run(predict_op, feed_dict={X: testX})))

			for startIndex, endIndex in zip( range(0,len(trainX),BATCH_SIZE), range(BATCH_SIZE,len(trainX),BATCH_SIZE)):
				sess.run(train_op, feed_dict={X: trainX[startIndex:endIndex], Y: trainY[startIndex:endIndex]})

		print "Training finished."
		print "==============================="


def main1():
	print "Loading the data......"
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	(nrTrainSamples, dimX) = trainX.shape
	(nrTestSamples, dimY)  = testY.shape

	print "Finished: data loaded. Stats below: "
	print "Nr of training samples: %d" %nrTrainSamples
	print "Nr of testing  samples: %d" %nrTestSamples
	print "Dimension of X: %d" %dimX
	print "Dimension of X: %d" %dimY
	IMAGE_DIM = int(dimX**0.5)  
	plt.imshow(np.reshape(trainX[0,:],(IMAGE_DIM,IMAGE_DIM)))


	# Build model
	X = tf.placeholder("float", [None, dimX])
	Y = tf.placeholder("float", [None, dimY])
	model = getModel(X, dimX, dimY)

	# Loss function
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

	# Optimizer
	train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss) 

	# Prediction/Output
	predict_op = tf.argmax(model, 1)
	# note: tf needs explicted casting after equal (don't need it in numpy): bool->float
	accuracy   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), predict_op), tf.float32))

	# Launch session for training
	trainModel(accuracy, train_op, X, Y, trainX, trainY, testX, testY)


if __name__ == '__main__':
	
	# (REPRODUCIBILITY) set random seeds
	# random.seed(123)
	tf.set_random_seed(123)
	
	main()