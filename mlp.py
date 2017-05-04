import tflearn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
	return X, Y

# Data loading and preprocessing
X, Y = load_data("data/max_temp.csv", training_size=365)

activation_func = 'leaky_relu'

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 11])
dense1 = tflearn.fully_connected(input_layer, 32, activation=activation_func,
                                 regularizer='L2')
dense2 = tflearn.fully_connected(dense1, 32, activation=activation_func,
                                 regularizer='L2')
dense3 = tflearn.fully_connected(dense2, 32, activation=activation_func,
                                 regularizer='L2')
dense4 = tflearn.fully_connected(dense3, 32, activation=activation_func,
                                 regularizer='L2')
dense5 = tflearn.fully_connected(dense4, 32, activation=activation_func,
                                 regularizer='L2')
output_layer = tflearn.fully_connected(dense5, 1, activation='linear')

# Regression using SGD + momentum with learning rate decay
sgd = tflearn.Momentum(learning_rate=0.001, lr_decay=0.96, decay_step=200)

net = tflearn.regression(output_layer, optimizer=sgd, loss='mean_square', metric='R2')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=1000, validation_set=0.2,
          show_metric=True, run_id="dense_model")

predictY = model.predict(X)

# Plot the results
steps_in_future = 1
plt.figure()
plt.title('Prediction of Max Temperature (degrees Celsius)')
plt.plot(Y, 'r-', label='Actual')
plt.plot(predictY, 'g-', label='Predicted')
plt.legend()
plt.show()
