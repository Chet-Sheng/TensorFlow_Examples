# ============================
# Tutorial - TF 
# (Check the load of dataset)
# -----------------------------
# Copyright @ dianaborsa
# =============================

import tensorflow as tf 
import numpy as np
import input_data
import matplotlib.pyplot as plt

def main():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	print trainX.shape
	print trainY.shape
	
	print 'Image (X): '
	plt.imshow(np.reshape(trainX[0,:],(28,28)))
	print 'Raw Y:' + str(trainY[0])
	print 'Label:' + str(np.argmax(trainY[0]))

if __name__ == '__main__':
	main()