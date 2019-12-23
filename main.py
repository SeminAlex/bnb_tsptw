import numpy as np
from os import listdir
from os.path import join
from itertools import product


def read_instance(filename):
    with open(filename, "r") as f:
        times = [0] + [int(v) for v in f.readline().split()]
        costs = list()
        for i in range(len(times)+1):
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
    def __init__(self, times, costs, opt_path=None, opt_value=None):
        self.costs = costs
        self.times = times
        self.opt_path = opt_path
        self.opt_value = opt_value

        self.n = len(times)
        self.full = set(range(1, self.n))

    def distance(self, path):
        dist = 0
        prev = 0
        for v in path:
            dist += self.costs[prev][v]
            prev = v
        return dist

    def cost_function(self, path):
        w = 0
        for i in range(1, len(path) + 1):
            if self.distance(path[:i]) > self.times[path[i-1]]:
                w += 1
        return w


class Vertex:
    def __init__(self, path=None):
        self.path = path

    def __add__(self, other):
        if other not in self.path:
            return Vertex(self.path + [other])
        return Vertex(self.path)

    def __str__(self):
        return "Vertex: Path = {}".format(self.path)

    def __repr__(self):
        return self.__str__()

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
    xkj = None
    for _ in range(task.n - len(vertex.path) - 1):
        beta = list(task.full - vertex.path_set)
        z = task.distance(vertex.path)
        last_vertex = vertex.path[-1]
        h = argmin(last_vertex, beta, z, task)
        xkj = beta[h]
        vertex = vertex + xkj
    if xkj:
        return task.cost_function(vertex.path+[xkj])
    else:
        return task.cost_function(vertex.path)


def lower_bound(vertex, task):
    beta = list(task.full - vertex.path_set)
    z = { vertex.path[i-1]: task.distance(vertex.path[:i]) for i in range(1, len(vertex.path) + 1)}

    vert_cost = task.distance(vertex.path)
    last_vert = vertex.path[-1]
    zb = {i: vert_cost + task.costs[last_vert][i] for i in beta}
    z.update(zb)
    return np.sum([0 if z[i] < task.times[i] else 1 for i in range(1, task.n)])


def brunching_wide(leafs):
    leafs = list(leafs)
    index = np.argmin([len(vert.path) for vert in leafs])
    return leafs[index]


def brunching_deep(leafs):
    leafs = list(leafs)
    index = np.argmax([len(vert.path) for vert in leafs])
    return leafs[index]


def brunch_and_bound(task, brunching, upper_calc, lower_calc):
    # step 1
    candidats = set()
    V = set([Vertex([i]) for i in range(1, task.n)])
    best_cost = None
    best_path = None
    while True:
        # step 3
        v = brunching(V)
        beta_all = task.full - v.path_set

        # step 3
        if len(v.path) == task.n - 1:
            candidats.add(v)

            cost = task.cost_function(v.path)
            dist = task.distance(v.path)
            if not best_cost or best_cost > cost:
                best_path = v
                best_cost = cost
                best_dist = dist
            print("\t\t", v, " COST = ", cost, "  DIST = ", dist)
        for beta in beta_all:
            V.add(v + beta)

        # step 4
        if len(beta_all) != 0:
            V.remove(v)
        remove = set()
        for v1, v2 in product(V, V):
            if v1 == v2:
                continue
            # print("BbB :: Check 1 :: ", v1, v2)
            # print("BbB :: Check 2 :: ", upper_calc(v1, task), lower_calc(v2, task))
            if v1 in V and v2 in V:
                lower = lower_calc(v2, task)
                if upper_calc(v1, task) <= lower or (best_cost and best_cost < lower):
                    if v2 in V:
                        V.remove(v2)
                        remove.add(v2)

        print("BnB Iteration : FOR REMOVE", remove)
        if v in V:
            V.remove(v)
        # for v in remove:
        #     V.remove(v)

        # step 2
        print("BnB Iteration : ", V, "\n\t\t SIZE = ", len(V))

        if len(V) == 0 or len(V) == 1 :
            print("BnB: Last step", candidats)
            cost = task.cost_function(best_path.path)
            dist = task.distance(best_path.path)
            print("\t\t", v, " COST = ", cost, "  DIST = ", dist)
            break
    return v, cost, dist


def main():
    instances_folder = "instances"
    for file in listdir(instances_folder):
        if file[-5:] != "5.txt":
            continue
        else:
            task = Task(*read_instance(join(instances_folder, file)))
            founded_path, founded_value = brunch_and_bound(task, brunching_deep, upper_bound, lower_bound)
            if founded_path == task.opt_path:  # and founded_value == task.opt_value:
                save_results(file, founded_path, founded_value)


if __name__ == "__main__":
    main()
