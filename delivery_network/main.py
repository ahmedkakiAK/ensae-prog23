import time
from graph import Graph, graph_from_file, min_power2
from statistics import mean

data_path = "input/"
file_name = "network.1.in"


def duration(i):
    g = graph_from_file(data_path + "network." + str(i) + ".in")
    route_file = 'input/routes.' + str(i) + '.in'
    times = []
    with open(route_file, "r") as file:
        n = int(file.readline())
        
        for j in range(5):
            src, dest, power = map(float, file.readline().split())
            t1 = time.perf_counter()
            p = g.min_power(src, dest)
            t2 = time.perf_counter()
            times.append(t2 - t1)

    return n * mean(times)


def duration2(i):
    g = graph_from_file(data_path + "network." + str(i) + ".in")
    route_file = 'input/routes.' + str(i) + '.in'
    times = []
    with open(route_file, "r") as file:
        out_file = 'input/routes.' + str(i) + '.out'
        file_power = open(out_file, "w")
        n = int(file.readline())
        
        for j in range(n-1):
            src, dest, power = map(float, file.readline().split())
            t1 = time.perf_counter()
            file_power.write(str(min_power2(g, src, dest)) + "\n")
            t2 = time.perf_counter()
            times.append(t2 - t1)

    return sum(times)


