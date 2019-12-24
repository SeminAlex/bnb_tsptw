import numpy as np
from os import listdir

from math import factorial
from itertools import product
from os.path import join, exists


def read_instance(filename):
    with open(filename, "r") as f:
        times = [0] + [int(v) for v in f.readline().split()]
        costs = list()
        for i in range(len(times)):
            costs.append([int(v) for v in f.readline().split()])
        opt_path = [int(v) for v in f.readline().split()]
        opt_value = [int(v) for v in f.readline().split()][0]
    return times, np.array(costs), opt_path, opt_value


def save_results(filename, path, value, count):
    total = factorial(len(path))
    # wrong variant
    complexity_wrong = (count - total) - total
    # write variant
    complexity_write = (count / total)
    with open(filename + "_out", "w") as f:
        for p in path:
            f.write(str(p) + " ")
        f.write("\nValue = " + str(value) + "\n")
        f.write("Complexity(original) = " + str(complexity_wrong))
        f.write("\nComplexity(custom) = " + str(complexity_write))


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
            if self.distance(path[:i]) > self.times[path[i - 1]]:
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
        return " ".join([str(i) for i in self.path])

    def __repr__(self):
        return "Vertex: Path = {}".format(self.path)

    def __iter__(self):
        return iter(self.path)

    def __len__(self):
        return len(self.path)

    def __hash__(self):
        result = 0
        for i in self.path:
            result *= 10 * len(str(i))
            result += i
        return result

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


def argmax(now, destines, z, task, ):
    def w_calc(dest):
        new_cost = task.times[dest] - z + task.costs[now][dest]
        if new_cost > 0:
            return new_cost
        return np.inf

    return np.argmax([w_calc(d) for d in destines])


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
        return task.cost_function(vertex.path + [xkj])
    else:
        return task.cost_function(vertex.path)


def my_upper_bound(vertex, task):
    for _ in range(task.n - len(vertex.path) - 1):
        beta = list(task.full - vertex.path_set)
        z = task.distance(vertex.path)
        last_vertex = vertex.path[-1]
        h = argmin(last_vertex, beta, z, task)
        xkj = beta[h]
        vertex = vertex + xkj
    # two-opt local search
    path = vertex.path.copy()
    current = task.cost_function(path)
    for i in range(task.n - len(vertex.path) - 2):
        for j in range(i, task.n - len(vertex.path) - 1):
            path[i], path[j] = path[j], path[i]
            if task.cost_function(path) > current:
                path[i], path[j] = path[j], path[i]
    return task.cost_function(path)


def lower_bound(vertex, task):
    beta = list(task.full - vertex.path_set)
    z = {vertex.path[i - 1]: task.distance(vertex.path[:i]) for i in range(1, len(vertex.path) + 1)}

    vert_cost = task.distance(vertex.path)
    last_vert = vertex.path[-1]
    zb = {i: vert_cost + task.costs[last_vert][i] for i in beta}
    z.update(zb)
    return np.sum([0 if z[i] < task.times[i] else 1 for i in range(1, task.n)])


def brunching_wide(leafs, *argc):
    leafs = list(leafs)
    index = np.argmin([len(vert.path) for vert in leafs])
    return leafs[index]


def brunching_deep(leafs, *argc):
    leafs = list(leafs)
    index = np.argmax([len(vert.path) for vert in leafs])
    return leafs[index]


def brunching_combine(leafs, flag=True, ):
    if flag:
        return brunching_deep(leafs)
    else:
        return brunching_wide(leafs)


def brunch_and_bound(task, brunching, upper_calc, lower_calc):
    viewed_vertexes = set()
    counter = 0
    # step 1
    V = set([Vertex([i]) for i in range(1, task.n)])
    best_cost = None
    best_path = None
    flag = True
    while True:
        # step 3
        v = brunching(V, flag)
        flag = not flag
        counter += 1
        beta_all = task.full - v.path_set

        # step 3
        if len(v.path) == task.n - 1:
            cost = task.cost_function(v.path)
            dist = task.distance(v.path)
            # flag = False
            if not best_cost or best_cost > cost:
                best_path = v
                best_cost = cost
                best_dist = dist
                print("CURRENT BEST COST ::", best_cost)
        for beta in beta_all:
            V.add(v + beta)

        # step 4
        V.remove(v)
        for v1, v2 in product(V, V):
            if v1 == v2:
                continue
            if v1 in V and v2 in V and v2 in V:
                lower = lower_calc(v2, task)
                if upper_calc(v1, task) <= lower or (best_cost and best_cost <= lower):
                    V.remove(v2)

        if len(V) == 0:
            cost = task.cost_function(best_path.path)
            dist = task.distance(best_path.path)
            print("\t\t", best_path, " COST = ", cost, "  DIST = ", dist)
            break
    return best_path, best_cost, best_dist, counter


def write_to_table(filename, instance, original, custom, target):
    template = "{instance};{value};{orig_val};{custom_val};{dist};{orig_dist};{custom_dist};{orig_count};{custom_count};{path};{orig_path};{custom_path}\n"
    if not exists(filename):
        writer = open(filename, 'w')
        writer.write(
            "Instance;Target Val;Orig BnB Val;Custom BnB Val;Target Dist;Orig BnB Dist;Custom BnB Dist;Orig BnB Count;Custom BnB Count;Target Path;Orig BnB Path;Custom BnB Path;\n")
    else:
        writer = open(filename, "a")
    writer.write(template.format(instance=instance.replace(".txt", ""), **original, **custom, **target))
    writer.close()


def main(bnb_algo=["origin", "custom"]):
    instances_folder = "instances"
    if not exists("results"):
        import os
        os.mkdir("results")
    if not exists("results_custom"):
        import os
        os.mkdir("results_custom")

    # result_table
    for file in listdir(instances_folder):
        if file[-4:] != ".txt":
            continue
        else:
            task = Task(*read_instance(join(instances_folder, file)))
            if "origin" in bnb_algo:
                founded_path, founded_value, dist, count = brunch_and_bound(task, brunching_deep, upper_bound,
                                                                            lower_bound)
                if founded_value == task.opt_value:
                    print("SAVE RESULTS FOR ORIGINAL BnB, Vertexes is looked ", count)
                    save_results(join("results", file), founded_path, founded_value, count)
                    original = {"orig_path": founded_path, "orig_val": founded_value, "orig_count": count,
                                "orig_dist": dist}
            if "custom" in bnb_algo:
                founded_path, founded_value, dist, count = brunch_and_bound(task, brunching_deep, my_upper_bound,
                                                                            lower_bound)
                if founded_value == task.opt_value:
                    print("SAVE RESULTS FOR CUSTOM BnB, Vertexes is looked ", count)
                    save_results(join("results_custom", file), founded_path, founded_value, count)
                    custom = {"custom_path": founded_path, "custom_val": founded_value, "custom_count": count,
                              "custom_dist": dist}
            cost = task.cost_function(task.opt_path)
            dist = task.distance(task.opt_path)
            print("\t\t", task.opt_path, " COST = ", cost, "  DIST = ", dist)
            if "custom" in bnb_algo and "origin" in bnb_algo:
                target = {"path":task.opt_path, "value":task.opt_value, "dist":task.cost_function(task.opt_path)}
                write_to_table("Results.csv", file, original, custom, target)


if __name__ == "__main__":
    import sys

    if len(sys.argv[1:]) > 0:
        main(sys.argv[1:])
    else:
        main()

