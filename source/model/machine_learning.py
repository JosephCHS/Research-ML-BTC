# System
import os
import string
import copy
# Pandas
import pandas
import pandas as pd
# Matplotlib
from matplotlib import pyplot, pyplot as plt
# Numpy
import numpy as np
from numpy import array, hstack
# Sklearn
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Tensorflow
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# Torch
import torchbnn as bnn
import torch
import torch.nn as nn
import torch.optim as optim
# Yellowbrick
from yellowbrick import ROCAUC
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ClassPredictionError


class MachineLearning:
    """This is a class used to create machine learning models to predict Bitcoin price.
    """

    def __init__(self, btc1=False):
        """Constructor method.

        :param btc1: boolean to determine if we have to used the data with btc1 2017-2021 (True)
        or without 2012-2021 (False)
        :type btc1: bool
        """
        np.random.seed(42)
        self.btc1 = btc1
        self.__csv_path = self.__get_absolut_path_to_data("../fetch/data/dataset.csv")
        if btc1 is True:
            self.__csv_path = self.__get_absolut_path_to_data("../fetch/data/dataset_with_future.csv")
        self.dataframe = self.get_dataframe()
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.dataframe,
                                                                                self.dataframe["Position"],
                                                                                test_size=0.3,
                                                                                random_state=42)
        return

    @staticmethod
    def __get_absolut_path_to_data(rel_path: string):
        """Returns the absolute path to the current file.

        :param rel_path: relative path to the current file
        :type rel_path: string
        :return: the absolute path to current file.
        :rtype: string
        """
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, rel_path)
        return abs_file_path

    def get_dataframe(self):
        """Returns the dataframe from the csv file.

        :return: Dataframe
        :rtype: pandas.DataFrame
        """
        # Get dataframe from csv
        dataframe = pd.read_csv(self.__csv_path)
        # Replace boolean by 0 and 1
        dataframe["Position"] = dataframe["Position"].astype(int)
        # Replace ? character used for Weka by NaN value
        dataframe = dataframe.replace("?", np.nan)
        # Replace NaN value by the previous value
        dataframe = dataframe.fillna(method='ffill')
        # Replace NaN value by the next value
        dataframe = dataframe.fillna(method='bfill')
        # Replace NaN value by 0
        dataframe = dataframe.fillna(0)
        # Convert date to datetime format
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        # Convert datetime to milliseconds from 01/01/1970
        dataframe["date"] = dataframe["date"].astype(np.int64) // 10 ** 9
        # If dataset with BTC1!CME remove useless attribute type
        if self.btc1 is True:
            dataframe = dataframe.drop(columns=['Value-BCDDY', 'low', "high", "Value-MKTCP", "close", "EMA"])
        return dataframe

    @staticmethod
    def display_result_regression(test_y, pred_y):
        """Display information about the prediction after a regression.
        """
        print(f"RMSE: {np.sqrt(metrics.mean_squared_error(test_y, pred_y))}")
        print(f"R2: {metrics.r2_score(test_y, pred_y)}")
        return

    @staticmethod
    def display_result_classification(test_y, pred_y):
        """Display information about the prediction after a classification.
        """
        print(f"Precision: {metrics.precision_score(test_y, pred_y)}")
        print(f"Recall: {metrics.recall_score(test_y, pred_y)}")
        print(f"Accuracy: {metrics.accuracy_score(test_y, pred_y)}")
        print(f"F1 score: {metrics.f1_score(test_y, pred_y, average=None)}")
        print(f"Confusion matrix: {metrics.confusion_matrix(test_y, pred_y, labels=[0, 1])}")
        return

    def display_information(self):
        """Display short information about the dataset.
        """
        print(f"The first five samples\n{self.train_x[:5]}.")
        print(f"The first five targets\n{self.train_y[:5]}.")
        print(f"The number of samples in train set is {self.train_x.shape[0]}.")
        print(f"The number of samples in test set is {self.test_x.shape[0]}.")
        return

    def display_rocauc(self, model):
        """Display ROC/AUC curve.

        :param model: relative path to the current file
        :type model: sklearn model
        """
        visualizer = ROCAUC(model, classes=["DOWN", "UP"])
        visualizer.fit(self.train_x, self.train_y)
        visualizer.score(self.test_x, self.test_y)
        visualizer.show()
        return

    def display_report_sklearn(self):
        """Display report for the sklearn models.
        """
        models = [
            LogisticRegression(solver='lbfgs'),
            svm.SVC(kernel="rbf")
        ]
        for model in models:
            # Report
            visualizer_report = ClassificationReport(model)
            visualizer_report.fit(self.train_x, self.train_y)
            visualizer_report.score(self.test_x, self.test_y)
            visualizer_report.show()
            # Prediction Error
            visualizer_error = ClassPredictionError(model)
            visualizer_error.fit(self.train_x, self.train_y)
            visualizer_error.score(self.test_x, self.test_y)
            visualizer_error.show()
        return

    def display_confusion_matrix(self, model):
        """Display the confusion matrix dynamically of the sklearn model.

        :param model: relative path to the current file
        :type model: model
        """
        cm = ConfusionMatrix(model, classes=["DOWN", "UP"])
        cm.fit(self.train_x, self.train_y)
        cm.score(self.test_x, self.test_y)
        cm.show()
        return

    def display_result(self, pred_y):
        """Returns the absolute path to the current file.

        :param pred_y: relative path to the current file
        :type pred_y: string
        """
        print(f"The first five prediction:\n{pred_y[:5]}.")
        print(f"The real first five labels:\n{self.test_y[:5]}.\n")
        print(f"Root Mean Squared Error:\n{np.sqrt(metrics.mean_squared_error(self.test_y, pred_y))}.")
        print(f"Confusion Matrix:\n{metrics.confusion_matrix(self.test_y, pred_y)}.")
        print(f"Precision: {metrics.precision_score(self.test_y, pred_y)}")
        print(f"Recall: {metrics.recall_score(self.test_y, pred_y)}")
        print(f"F1 score: {metrics.f1_score(self.test_y, pred_y, average=None)}")
        print(f"Accuracy Score: {metrics.accuracy_score(y_true=self.test_y, y_pred=pred_y)}.")
        return

    @staticmethod
    def draw_graph(pred_y, test_x, test_y):
        """Returns the absolute path to the current file.

        :param pred_y: prediction made by the model
        :type pred_y: [[]]
        :param test_x: data used to test the model
        :type test_x: [[]]
        :param test_y: label used to test the data in test_x
        :type test_y: [[]]
        """
        fig = plt.figure(figsize=(16, 8))
        fig_1 = fig.add_subplot(1, 2, 1)
        fig_2 = fig.add_subplot(1, 2, 2)
        z1_plot = fig_1.scatter(test_x[:, 0], test_x[:, 1], c=test_y, marker="v")
        z2_plot = fig_2.scatter(test_x[:, 0], test_x[:, 1], c=pred_y)
        plt.colorbar(z1_plot, ax=fig_1)
        plt.colorbar(z2_plot, ax=fig_2)
        fig_1.set_title("REAL")
        fig_2.set_title("PREDICT")
        plt.show()
        return

    @staticmethod
    def normalise_dataframe(dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """Normalise the dataset by value between 0 and 1.

        :param dataframe: relative path to the current file
        :type dataframe: pandas.DataFrame
        :return: normalised dataframe
        :rtype: pandas.DataFrame
        """
        min_max = MinMaxScaler()
        x_scaled = min_max.fit_transform(dataframe)
        dataframe = pd.DataFrame(x_scaled, columns=dataframe.columns)
        return dataframe

    @staticmethod
    def regression_to_classification(pred_y):
        """Discretise the pred_y array, if the value is close to 1 than 0 then the value become 0 and vise versa for 0.

        :param pred_y: relative path to the current file
        :type pred_y: pandas.DataFrame
        :return: pred_y value discretised
        :rtype: [[]]
        """
        for idx in pred_y:
            if idx[0] > 0.5:
                idx[0] = 1
            else:
                idx[0] = 0
        return pred_y

    @staticmethod
    def to_sequences(seq_size, data, close):
        """Adapt the 2D array used as feature and label with the seq_size.

        :param seq_size: integer, sequence size
        :type seq_size: int
        :param data: feature used for training
        :type data: [[]]
        :param close: label used for supervised training
        :type close: [[]]
        :return: Two 2D array resized with the seq_size
        :rtype: [[]], [[]]
        """
        x = []
        y = []
        for index in range(len(data) - seq_size - 1):
            window = data[index:(index + seq_size)]
            after_window = close[index + seq_size]
            window = [[x] for x in window]
            x.append(window)
            y.append(after_window)
        return np.array(x), np.array(y)

    def model_lstm(self):
        """Create a LSTM model to do prediction over a dataset. And display its results. Keras is used.
        This LSTM is based on the architecture available in:
            Article: Multivariate Time Series Forecasting with LSTMs in Keras
            Author: Jason Brownlee
            Date: August 14, 2017
            Book: Deep Learning for Time Series
            URL: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        """
        dataframe = self.get_dataframe()
        dataframe = self.normalise_dataframe(dataframe)
        dataframe = dataframe.astype(np.float32)
        df_list = []
        position = array([])
        for column in dataframe:
            if dataframe[column].name == "Position":
                position = dataframe[column].to_numpy()
                position = position.reshape((len(position), 1))
                continue
            tmp_data = dataframe[column].to_numpy()
            df_list.append(tmp_data.reshape((len(tmp_data), 1)))
        df_list.append(position)
        array_np = tuple(df_list)
        dataset = hstack(array_np)
        ## Split the dataset 70/30: 2200 FULL // 915 BTC1!
        train = dataset[:2200, :]
        test = dataset[2200:, :]
        # Split into input and outputs
        train_x, train_y = train[:, :-1], train[:, -1]
        test_x, test_y = test[:, :-1], test[:, -1]
        # Reshape input to be 3D [samples, timesteps, features]
        train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
        # Create LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # Fit network
        history = model.fit(train_x, train_y,
                            epochs=60, batch_size=120,
                            validation_data=(test_x, test_y),
                            verbose=0, shuffle=False)
        # Plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        # Make a prediction
        pred_y = model.predict(test_x)
        self.display_result_regression(test_y, pred_y)
        pred_y = self.regression_to_classification(pred_y)
        self.display_result_classification(test_y, pred_y)
        return pred_y

    def model_cnn(self):
        """Create a CNN model to do prediction over a dataset. And display its results. Keras is used.
        This CNN is based on the architecture available in:
            Article: Multivariate Time Series Forecasting with LSTMs in Keras
            Author: Salman Chen
            Date: May 8, 2020
            URL: https://www.kaggle.com/salmanhiro/stock-closing-price-prediction-with-cnn
        """
        dataframe = self.get_dataframe()
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataframe = scaler.fit_transform(dataframe)
        dataframe = dataframe.astype(np.float32)
        close_y = dataframe[:, 13]
        # Split the dataset 70/30: 2200 FULL // 915 BTC1!
        ntrain = 2200
        if self.btc1 is True:
            ntrain = 915
        train = dataframe[0:ntrain]
        test = dataframe[ntrain:len(dataframe)]
        train_close_y = close_y[0:ntrain]
        test_close_y = close_y[ntrain:len(close_y)]
        timesteps = 50
        train_x, train_y = self.to_sequences(timesteps, train, train_close_y)
        x_test, y_test = self.to_sequences(timesteps, test, test_close_y)
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[2], train_x.shape[1], train_x.shape[3]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2], x_test.shape[1], x_test.shape[3]))
        # Display 3 first images of the dataset
        fig, (ax1, ax2, ax3) = pyplot.subplots(1, 3)
        ax1.imshow(train_x[0][0])
        ax2.imshow(train_x[1][0])
        ax3.imshow(train_x[2][0])
        pyplot.figure(num=1, figsize=(10, 10))
        fig.show()
        # CNN
        cnn = Sequential()
        # Input shape 1 50 27/21
        nb_feature = 27
        if self.btc1 is True:
            nb_feature = 21
        cnn.add(Conv2D(8, kernel_size=(1, 6), strides=(1, 1), padding='valid',
                       activation='relu', input_shape=(1, 50, nb_feature)))
        cnn.add(MaxPooling2D(pool_size=(1, 6)))
        cnn.add(Flatten())
        cnn.add(Dense(4, activation="relu"))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(1, activation="relu"))
        cnn.summary()
        cnn.compile(loss='mean_squared_error', optimizer='nadam')
        EarlyStopping(monitor='val_loss', min_delta=1, patience=2, verbose=2, mode='auto')
        # Save checkpoint weights
        checkpointer = ModelCheckpoint(filepath="results/parameters.hdf5", verbose=0, save_best_only=True)
        # Display curve loss during training
        history = cnn.fit(train_x, train_y, validation_split=0.2,
                          batch_size=126, callbacks=[checkpointer],
                          verbose=0, epochs=60)
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        cnn.load_weights('results/parameters.hdf5')
        # Prediction
        pred_y = cnn.predict(x_test)
        # Display results
        self.display_result_regression(y_test, pred_y)
        pred_y = self.regression_to_classification(pred_y)
        print(pred_y.shape)
        self.display_result_classification(y_test, pred_y)
        return pred_y

    def model_bnn(self):
        """Create a BNN model to do prediction over a dataset. And display its results. Pytorch is used.
        This BNN is based on the architecture available in:
            Package: https://pypi.org/project/torchbnn/
            Documentation:
            Author: Harry Kim
            Date: June 18, 2020
        """
        dataframe = self.get_dataframe()
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataframe = scaler.fit_transform(dataframe)
        dataframe = dataframe.astype(np.float32)
        close_y = dataframe[:, 13]
        ## Split the dataset 70/30: 2200 FULL // 915 BTC1!
        ntrain = 2200
        if self.btc1 is True:
            ntrain = 915
        # Data BNN
        data = dataframe[ntrain:len(dataframe)]
        target = close_y[ntrain:len(close_y)]
        test_x = torch.from_numpy(data).float()
        test_y = torch.from_numpy(target).long()
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()
        nb_feature = 27
        if self.btc1 is True:
            nb_feature = 21
        # Model BNN
        model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=nb_feature, out_features=100),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=2),
        )
        # Loss functions
        cross_entropy_loss = nn.CrossEntropyLoss()
        kl_div_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.01
        adam_optimiser = optim.Adam(model.parameters(), lr=0.01)
        # Training
        cross_entropy = 0
        kl_loss = 0
        models = 0
        for step in range(10):
            models = model(test_x)
            cross_entropy = cross_entropy_loss(models, target)
            kl_loss = kl_div_loss(model)
            total_cost = cross_entropy + kl_weight * kl_loss
            adam_optimiser.zero_grad()
            total_cost.backward()
            adam_optimiser.step()
        # Prediction
        _, pred_y = torch.max(models.data, 1)
        final = test_y.size(0)
        correct = (pred_y == test_y).sum()
        print(f"Accuracy: {(100 * float(correct) / final)}%%")
        print(f"Cross-entropy: {cross_entropy.item()}")
        print(f"Kullback-Leibler loss: {kl_loss.item()}")
        # Display results
        self.display_result_classification(test_y, pred_y)
        self.display_result_regression(test_y, pred_y)
        models = model(data)
        _, pred_y = torch.max(models.data, 1)
        self.draw_graph(pred_y, data, target)
        print(pred_y)
        return

    def model_logistic_regression(self):
        """Create a LR model to do prediction over a dataset. And display its results. Sklearn is used.
        """
        classifier = LogisticRegression(solver='lbfgs')
        self.display_rocauc(copy.deepcopy(classifier))
        self.display_confusion_matrix(copy.deepcopy(classifier))
        classifier.fit(self.train_x, self.train_y)
        pred_y = classifier.predict(self.test_x)
        return pred_y

    def model_svm(self):
        """Create a SVM model to do prediction over a dataset. And display its results. Sklearn is used.
        """
        classifier = svm.SVC(kernel="rbf")
        self.display_confusion_matrix(copy.deepcopy(classifier))
        classifier.fit(self.train_x, self.train_y)
        pred_y = classifier.predict(self.test_x)
        return pred_y


def main():
    # Instantiate class
    machine_learning = MachineLearning()
    machine_learning.display_information()
    # Model sklearn
    models_sklearn = [
        machine_learning.model_logistic_regression(),
        machine_learning.model_svm(),
    ]
    # Display models results sklearn
    for model in models_sklearn:
        machine_learning.display_result(model)
    machine_learning.display_report_sklearn()
    # Model Keras
    machine_learning.model_lstm()
    machine_learning.model_cnn()
    # Model Pytorch
    machine_learning.model_bnn()
    return


if __name__ == '__main__':
    main()
