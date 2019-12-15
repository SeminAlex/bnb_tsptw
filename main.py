import numpy as np
from os import listdir
from os.path import join


def read_instance(filename):
    with open(filename, "r") as f:
        times = f.readline().split()
        costs = list()
        for i in range(len(times)):
            costs.append([int(v) for v in f.readline().split()])
        opt_path = [int(v) for v in f.readline().split()]
        opt_value = [int(v) for v in f.readline().split()]
    return times, costs, opt_path, opt_value


def save_results(filename, path, value):
    with open(filename + "_out", "w") as f:
        for p in path:
            f.write(str(p) + " ")
        f.write("\n" + str(value) + "\n")


class Task:
    def __init__(self, costs, times, opt_path=None, opt_value=None):
        self.costs = costs
        self.times = times
        self.opt_path = opt_path
        self.opt_value = opt_value

        self.n = len(times)
        self.full = set(range(self.n))

    def distance(self, path):
        dist = 0
        prev = 0
        for v in path:
            dist += self.costs[prev][v]
            prev = v
        return dist


class Vertex:
    def __init__(self, path=None):
        self.path = path

    def __add__(self, other):
        if other not in self.path:
            return Vertex(self.path + [other])
        return Vertex(self.path)

    @property
    def path_set(self):
        return set(self.path)


def argmin(now, destines, z, task, ):
    def w_calc(dest):
        new_cost = task.times[dest] - z + task.costs[now][dest]
        if new_cost > 0:
            return new_cost
        return np.inf

    return np.argmin([w_calc(d) for d in destines])


def upper_bound(vertex, task):
    beta = list(task.full - vertex.path_set)

    if len(beta) != 0:
        z = task.distance(vertex.path)
        h = argmin(vertex.path[-1], beta, z, task)
        xkj = beta[h]
        if len(beta) > 1:
            return upper_bound(vertex + xkj, task)
        else:
            return z + task.costs[vertex.path[-1]][xkj]
    else:
        return task.distance(vertex.path)


def lower_bound(vertex, task):
    beta = list(task.full - vertex.path_set)
    za = [task.distance(vertex.path[:i]) for i in range(1, len(vertex.path))]
    vert_cost = task.distance(vertex.path)
    last_vert = vertex.path[-1]
    zb = [vert_cost + task.costs[last_vert][i] for i in beta]
    z = za + zb
    return np.sum([0 if z[i] < task.times[i] else 1 for i in range(task.n)])


def brunching_wide(leafs):
    leafs = list(leafs)
    index = np.argmin([len(vert.path) for vert in leafs])
    return leafs[index]


def brunching_deep(leafs):
    leafs = list(leafs)
    index = np.argmax([len(vert.path) for vert in leafs])
    return leafs[index]


def brunch_and_bound(task, brunching, upper_calc, lower_calc):
    return None, None


def main():
    instances_folder = "instances"
    for file in listdir(instances_folder):
        if file[-4:] != ".txt":
            continue
        else:
            task = Task(*read_instance(join(instances_folder, file)))
            founded_path, founded_value = brunch_and_bound(task, brunching_deep, upper_bound, lower_bound)
            if founded_path == task.opt_path and founded_value == task.opt_value:
                save_results(file, founded_path, founded_value)
