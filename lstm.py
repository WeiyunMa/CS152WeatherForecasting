import tflearn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

steps_of_history = 200
training_size = 365 * 4 - 200

def load_data2(path, window_size, training_size):
    weather_data = pd.read_csv(path, sep='\t')

    test_list = []
    for year in range(2005, 2010):
        for j in range(0, 365):
            test_list.append(weather_data[str(year)][j])

    X = []
    Y = []
    for index in range(0, len(test_list) - window_size):
        xx = []
        for j in range(index, index + window_size):
            xx.append([test_list[j]])
        X.append(xx)
        Y.append([test_list[index + window_size]])
    X = np.array(X)
    Y = np.array(Y)
    trainX, trainY = X[:training_size], Y[:training_size]
    testX, testY = X[training_size:], Y[training_size:]
    print(trainX.shape)
    print(testY.shape)
    return trainX, trainY, testX, testY, X, Y

# Data loading and preprocessing
trainX, trainY, testX, testY, X, Y = load_data2("data/max_temp.csv",
                                          window_size=steps_of_history,
                                          training_size=training_size)

# Network building
net = tflearn.input_data([None, steps_of_history, 1])
# net = tflearn.lstm(net, n_units=128, activation='tanh', return_seq=True)
# net = tflearn.lstm(net, n_units=128, activation='tanh', return_seq=True)
net = tflearn.lstm(net, n_units=128, activation='tanh', return_seq=False)
net = tflearn.fully_connected(net, 1, activation='linear')
sgd = tflearn.Momentum(learning_rate=0.001, lr_decay=0.96, decay_step=200)
net = tflearn.regression(net, optimizer=sgd, loss='mean_square', learning_rate=0.001)

# Training
model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=200, validation_set=0.1, batch_size=128)

# Predict the future values
predictY = model.predict(X)

# Plot the results
steps_in_future = 1
plt.figure()
plt.suptitle('Prediction of Max Temperature (degrees Celsius)')
plt.title('History window size ='+str(steps_of_history)+', Future step ='+str(steps_in_future))
plt.plot(Y, 'r-', label='Actual')
plt.axvline(x=training_size, ls='dashed', color='b')
plt.plot(predictY, 'g-', label='Predicted')
plt.legend()
plt.show()
