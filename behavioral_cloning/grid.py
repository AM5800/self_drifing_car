import copy
import os
import pickle


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


class GridManager:
    def __init__(self, file_name):
        self.__grids = {}
        self.__file_name = file_name
        self.__results = {}
        self.__best_result_value = 1000

        if os.path.isfile(file_name):
            data = pickle.load(open(file_name, "rb"))
            self.__results = data["results"]
            self.__best_result_value = data["best_result_value"]

    def add(self, grid_id, param_name, param_value):
        new_grid = []
        for node in self.__grids.setdefault(grid_id, [{}]):
            for value in param_value:
                node_copy = copy.deepcopy(node)
                node_copy[param_name] = value
                new_grid.append(node_copy)

        self.__grids[grid_id] = new_grid

    def save_node_result(self, node, result):
        self.__results[self.__node_to_hashable(node)] = (node, result)
        self.save()

    def save(self):
        pickle.dump(
            {
                "results": self.__results,
                "best_result_value": self.__best_result_value
            }, open(self.__file_name, "wb"))

    def submit_best_result_value(self, best_result):
        self.__best_result_value = min(self.__best_result_value, best_result)
        self.save()

    def get_best_result_value(self):
        return self.__best_result_value

    @staticmethod
    def __node_to_hashable(node):
        return str(sorted(node.items()))

    def get_new_nodes(self):
        result = []

        tested_nodes = set(self.__results.keys())

        for node in flatten(self.__grids.values()):
            node_str = self.__node_to_hashable(node)
            if node_str not in tested_nodes:
                result.append(node)

        return result

    def get_results(self):
        return list(self.__results.values())

    def new_grid(self):
        return len(self.__grids)
