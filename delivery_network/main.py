import time
from graph import Graph, graph_from_file
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


def time_explo(x):
    g = graph_from_file('input/network.'+str(x)+'.in')
    route_file = 'input/routes.'+ str(x) + '.in'

    with open(route_file, "r") as file:
        n = file.readline().split()
        n = int(n[0])
        lines =file.readlines()
        l_time = []
        for line in lines[0:5] :
            line = line.split()
            src, dest, weight = map(float,line) # Choice of parameters
            t0 = time.perf_counter()
            Graph.min_power(g, src, dest)
            t1 = time.perf_counter()
            t = t1 - t0  # difference between the times before and after execution of the function
            l_time.append(t)

        print('time:' + str(mean(l_time))*n + ' secondes')
    
time_explo(2)



