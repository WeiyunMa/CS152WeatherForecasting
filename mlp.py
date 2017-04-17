import tflearn
import pandas as pd
import numpy as np

def load_data(path, training_size):
	weather_data = pd.read_csv(path, sep='\t')
	X = []
	Y = []
	for index, row in weather_data.iterrows():
		if index > 0:
			Y.append([row['2009']])
			input = []
			for year in range(1999,2009):
				input.append(row[str(year)])
			input.append(yesterday)
			X.append(input)
		yesterday = row['2009']
	X = np.array(X)
	Y = np.array(Y)
	trainX, trainY = X[:training_size], Y[:training_size]
	testX, testY = X[training_size:], Y[training_size:]
	# print(trainX.shape, trainY.shape, testX.shape, testY.shape)
	# print(trainX[0])
	# print(trainY[0])
	return trainX, trainY, testX, testY

# Data loading and preprocessing
trainX, trainY, testX, testY = load_data("data/max_temp.csv", training_size=300)

trainX = trainX + trainX
trainY = trainY + trainY

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 11])
dense1 = tflearn.fully_connected(input_layer, 16, activation='tanh',
                                 regularizer='L2')
#dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dense1, 16, activation='tanh',
                                 regularizer='L2')
#dropout2 = tflearn.dropout(dense2, 0.8)
dense3 = tflearn.fully_connected(dense2, 16, activation='tanh',
                                 regularizer='L2')
#dropout3 = tflearn.dropout(dense3, 0.8)
dense4 = tflearn.fully_connected(dense3, 16, activation='tanh',
                                 regularizer='L2')
#sdropout4 = tflearn.dropout(dense4, 0.8)
dense5 = tflearn.fully_connected(dense4, 16, activation='tanh',
                                 regularizer='L2')
softmax = tflearn.fully_connected(dense5, 1, activation='softmax')

# Regression using SGD with learning rate decay
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)

net = tflearn.regression(softmax, optimizer=sgd, loss='mean_square')

# Training
model = tflearn.DNN(net, tensorboard_verbose=3)

model.fit(trainX, trainY, n_epoch=1000, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")