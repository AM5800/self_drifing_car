import copy
import os
import pickle


class GridManager:
    def __init__(self, file_name):
        self.__grid = [{}]
        self.__file_name = file_name
        self.__results = {}
        self.__best_result_value = 1000

        if os.path.isfile(file_name):
            self.__results = pickle.load(open(file_name, "rb"))
            self.__best_result_value = max(self.__results, key=lambda x: x[1][0])

    def add(self, name, values):
        new_grid = []
        for node in self.__grid:
            for value in values:
                node_copy = copy.deepcopy(node)
                node_copy[name] = value
                new_grid.append(node_copy)

        self.__grid = new_grid

    def submit(self, result_value, node, result):
        self.__results[self.__node_to_hashable(node)] = (result_value, result)
        self.__best_result_value = min(self.__best_result_value, result_value)

        pickle.dump(self.__results, open(self.__file_name, "wb"))

    def get_best_result_value(self):
        return self.__best_result_value

    @staticmethod
    def __node_to_hashable(node):
        return str(sorted(node.items()))

    def get_new_nodes(self):
        result = []

        tested_nodes = set(self.__results.keys())

        for node in self.__grid:
            node_str = self.__node_to_hashable(node)
            if node_str not in tested_nodes:
                result.append(node)

        return result
