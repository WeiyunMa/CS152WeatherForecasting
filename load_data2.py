import pandas as pd


def load_data2(path, window_size):
    weather_data = pd.read_csv(path, sep='\t')
    weather_biglist = []
    # for year in range(1999, 2010):
        # for j in range(0, 366):
            # weather_biglist.append(weather_frame[str(year)][j])
    test_list = []
    for year in range(2007, 2010):
        for j in range(0, 366):
            test_list.append(weather_frame[str(year)][j])

    X = []
    Y = []
    for index in range(0, len(test_list) - window_size):
        X.append(test_list[index:index+window_size])
        Y.append(test_list[window_size])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y





