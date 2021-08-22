import os
import string

import pandas as pd


class Arff:
    """This is a class used to convert csv file to arff file for data mining with Weka
    """
    def __init__(self):
        """Constructor method
        """
        self.__header_path = self.__get_absolut_path_to_data("data/arff_header.txt")
        self.__csv_path = self.__get_absolut_path_to_data("../fetch/data/dataset.csv")
        self.__weka_path = self.__get_absolut_path_to_data("data/weka.arff")
        self.__header_with_future_path = self.__get_absolut_path_to_data("data/arff_header.txt")
        self.__csv_with_future_path = self.__get_absolut_path_to_data("../fetch/data/dataset_with_future.csv")
        self.__weka_with_future_path = self.__get_absolut_path_to_data("data/weka_with_future.arff")
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

    def generate_header(self):
        header = str()
        with open(self.__header_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip("\n") == "@ATTRIBUTE Close-future REAL":
                continue
            header += line
        return header

    def generate_header_with_future(self):
        with open(self.__header_path, "r") as file:
            return file.read()

    @staticmethod
    def remove_first_line_file(path: string):
        with open(path, 'r') as file:
            data = file.read().splitlines(True)
        return data[1:]

    def generate_arff(self):
        csv = self.remove_first_line_file(self.__csv_path)
        header = self.generate_header()
        csv.insert(0, header)
        with open(self.__weka_path, 'w') as arff:
            arff.write(" ".join(csv))
        return

    def generate_arff_with_future(self):
        csv = self.remove_first_line_file(self.__csv_with_future_path)
        header = self.generate_header_with_future()
        csv.insert(0, header)
        with open(self.__weka_with_future_path, 'w') as arff:
            arff.write(" ".join(csv))
        return


#def main():
#    convert = Arff()
#    convert.generate_arff()
#    convert.generate_arff_with_future()
#    return


#if __name__ == '__main__':
#    main()
