import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from numpy import NaN


class ConfusionMatrix:
    """This is a class used to create confusion matrix
    """

    def __init__(self, default_results=False, lr=NaN, svm=NaN, lstm=NaN, cnn=NaN, bnn=NaN):
        """Constructor method
        Returns the absolute path to the current file.

        :param default_results: if true, used the results for the essay
        :type default_results: bool
        :param lr: 2D array with the confusion matrix for LR
        :type lr: [[]]
        :param svm: 2D array with the confusion matrix for LR
        :type svm: [[]]
        :param lstm: 2D array with the confusion matrix for LR
        :type lstm: [[]]
        :param cnn: 2D array with the confusion matrix for LR
        :type cnn: [[]]
        :param bnn: 2D array with the confusion matrix for LR
        :type bnn: [[]]
        """
        self.lr = lr
        self.svm = svm
        self.lstm = lstm
        self.cnn = cnn
        self.bnn = bnn
        self.__set_default_results(default_results)
        self.classes = ["DOWN", "UP"]
        self.models = {"BNN": self.bnn,
                       "CNN": self.cnn,
                       "LSTM": self.lstm,
                       "SVM": self.svm,
                       "LR": self.lr
                       }
        self.__display_matrix()
        return

    def __set_default_results(self, default_results: bool):
        """Returns the results used for the essay.

        :param default_results: if set to true the default results are used
        :type default_results: bool
        """
        if default_results is True:
            self.svm = [[68, 359],
                        [80, 492]]
            self.lr = [[113, 314],
                       [159, 413]]
            self.lstm = [[223, 299],
                         [160, 428]]
            self.cnn = [[66, 441],
                        [61, 511]]
            self.bnn = [[17, 515],
                        [0, 598]]
        return

    def __display_matrix(self):
        """Display the confusion matrix for the five models.
        """
        for key, value in self.models.items():
            df_cfm = pd.DataFrame(value, index=self.classes, columns=self.classes)
            plt.figure(figsize=(10, 7))
            cfm_plot = sn.heatmap(df_cfm, annot=True, cmap='YlGnBu', fmt='g')
            cfm_plot.set_xlabel("Predicted Class")
            cfm_plot.set_ylabel("True Class")
            cfm_plot.figure.savefig(f"Results/{key},no-f,CM.png")
        return


#def main():
#    ConfusionMatrix(True)
#    return


#if __name__ == '__main__':
#    main()
