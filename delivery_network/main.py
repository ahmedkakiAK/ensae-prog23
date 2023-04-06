import time
from graph import Graph, graph_from_file, min_power2, depths_nodes, kruskal, lca, binary_lifting, oriented_tree
from statistics import mean


data_path = "input/"
file_name = "network.1.in"
B = 25 * 10**9 #Budget

def duration(i): # Fonction permettant de trouver le temps d'éxecution de min_power pour un graphe correspondant à network.i.in
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


def duration2(i): # Fonction permettant de calculer le temps d'exécution de min_power2 pour network.i.in et de stocker les puissances minimales dans un fichier
    g = graph_from_file(data_path + "network." + str(i) + ".in")
    route_file = 'input/routes.' + str(i) + '.in'
    times = []
    with open(route_file, "r") as file:
        out_file = 'input/routes.' + str(i) + '.out'
        file_power = open(out_file, "w")
        n = int(file.readline())
        
        for j in range(n-1):
            src, dest, utility = map(int, file.readline().split())
            t1 = time.perf_counter()
            file_power.write(str(src) + " " + str(dest) + " " + str(utility) + " " + str(min_power2(g, src, dest)) + "\n")
            t2 = time.perf_counter()
            times.append(t2 - t1)

    return sum(times)



def trucks_tri(i):
    filename = data_path + 'trucks' + str(i) + '.in'
    with open(filename, 'r') as file:
        nb_trucks = int(file.readline)
        trucks = []
        for j in range(nb_trucks):
            power, cost = map(int, file.readline().split())
            trucks.append((power, cost))
    trucks.sort(key=lambda x: (x[0],-x[1]))
    trucks=[trucks[-1]]
    for elt in trucks[-2::-1]:
        if elt[1]<trucks[-1][1]:
            trucks.append(elt)
    return trucks[::-1]


def pre_knapsack(i):
    filename = 'input/routes.' + str(i) + '.out'
    trucks = trucks_tri(i)
    with open(filename, 'r') as file:
        nb_trajets = int(file.readline())
        trajets = []
        for _ in range(nb_trajets):
            src, dest, utility, power = map(int, file.readline().split())
            trajets.append(src, dest, utility, power)
    trajets.sort(key=lambda x: (x[3]))
    knapsack = []
    content = 0
    nb_trucks = len(trucks)
    for j in range(nb_trucks):
        if trucks[j][0]>=trajets[i][3]:
            while i<len(trajets) and trucks[j][0]>=trajets[i][3] :
                knapsack.append((trucks[j][0],trucks[j][1],trajets[i][2],trajets[i][0],trajets[i][1])) #Puissance, coût, profit, src, dest
                i+=1
        if i>=len(trajets):
            return knapsack
        

"""def sol_knapsack(knapsack):
    matrice = {0:np.zeros(B+1,dtype=int)}
    for i in range(1, len(knapsack) + 1):
        for w in range(0, B+1):
            print(w)
            if knapsack[i-1][1] <= w:
                matrice[i].append(max(knapsack[i-1][2] + matrice[i-1][w-knapsack[i-1][1]], matrice[i-1][w]))
            else:
                matrice[i].append(matrice[i-1][w])
            
    w = B
    n = len(knapsack)
    content_bag = []

    while w >= 0 and n >= 0:
        e = knapsack[n-1]
        if matrice[n][w] == matrice[n-1][w-e[1]] + e[2]:
            content_bag.append(e)
            w -= e[1]

        n -= 1

    return content_bag, matrice[-1][-1]"""


def greedy(knapsack):
    knapsack.sort(key=lambda x: x[1]/x[2])
    bag = [] 
    k=0
    S=0
    P=0
    n = len(knapsack)
    while k < n:
        if S + knapsack[k][1]  <=B:
            bag.append((knapsack[k][0], knapsack[k][1] , knapsack[k][3], knapsack[k][4]))
            S += knapsack[k][1]
            P += knapsack[k][2]
            k += 1
        else:
            return bag ,S
    return bag ,S


file_name1 = "network.1.in"
g1= graph_from_file(data_path+file_name1)
mst1=kruskal(g1)
T1=oriented_tree(mst1,1)
D1=depths_nodes(mst1)
Ancestors1=binary_lifting(T1)

file_name2 = "network.2.in"
g2 = graph_from_file(data_path+file_name2)
mst2=kruskal(g2)
T2=oriented_tree(mst2,1)
D2=depths_nodes(mst2)
Ancestors2=binary_lifting(T2)

file_name3 = "network.3.in"
g3 = graph_from_file(data_path+file_name3)
mst3=kruskal(g3)
T3=oriented_tree(mst3,1)
D3=depths_nodes(mst3)
Ancestors3=binary_lifting(T3)

file_name4 = "network.4.in"
g4 = graph_from_file(data_path+file_name4)
mst4=kruskal(g4)
T4=oriented_tree(mst4,1)
D4=depths_nodes(mst4)
Ancestors4=binary_lifting(T4)

file_name5 = "network.5.in"
g5= graph_from_file(data_path+file_name5)
mst5=kruskal(g5)
T5=oriented_tree(mst5,1)
D5=depths_nodes(mst5)
Ancestors5=binary_lifting(T5)

file_name6 = "network.6.in"
g6 = graph_from_file(data_path+file_name6)
mst6=kruskal(g6)
T6=oriented_tree(mst6,1)
D6=depths_nodes(mst6)
Ancestors6=binary_lifting(T6)

file_name7 = "network.7.in"
g7 = graph_from_file(data_path+file_name7)
mst7=kruskal(g7)
T7=oriented_tree(mst7,1)
D7=depths_nodes(mst7)
Ancestors7=binary_lifting(T7)

file_name8 = "network.8.in"
g8 = graph_from_file(data_path+file_name8)
mst8=kruskal(g8)
T8=oriented_tree(mst8,1)
D8=depths_nodes(mst8)
Ancestors8=binary_lifting(T8)

file_name9 = "network.9.in"
g9 = graph_from_file(data_path+file_name9)
mst9=kruskal(g9)
T9=oriented_tree(mst9,1)
D9=depths_nodes(mst9)
Ancestors9=binary_lifting(T9)

file_name10 = "network.10.in"
g10 = graph_from_file(data_path+file_name10)
mst10=kruskal(g10)
T10=oriented_tree(mst10,1)
D10=depths_nodes(mst10)
Ancestors10=binary_lifting(T10)




def min_power_tree(tree,src,dest):
    """""   Fonction : min_power_tree
    Description:
    -----------
    trouve le chemin entre deux noeuds dans un arbre en remontant les ancêtres pour touver 
    l'ancêtre minimum. On peut utiliser la profondeur des noeuds pour aller plus vite mais c'est déjà assez
    rapide sans. On utilise la profondeur dans la séance 3 où le même principe est repris
    pour aller justement beaucoup plus vite.
    Input:
    ------
    src : noeud de départ
    type : int
    dest : noeud d'arrivé
    type : int
    tree : arbre 
    output : 
    -------
    un couple contenant la puissance minimal et le chemin de src à dest
    Complexity : O(N) dans le pire cas. On se retrouve à parcourir tous les sommets
    """
    # Remonter les ancêtres de src
    src_ancestors = []
    curr = src
    while curr!=1: #CHOIX ARBITRAIRE DE LA RACINE COMME ÉTANT 1
        src_ancestors.append([curr,tree[curr][0][1]])
        curr = tree[curr][0][0]
    src_ancestors.append([1,0])
    # Remonter les ancêtres de dest
    dest_ancestors = []
    curr = dest
    while curr!=1: #CHOIX ARBITRAIRE DE LA RACINE COMME ÉTANT 1
        dest_ancestors.append([curr,tree[curr][0][1]])
        curr = tree[curr][0][0]
    dest_ancestors.append([1,0])
    # Trouver l'indice du premier ancêtre commun entre src et dest. On peut utiliser la profondeur pour éviter de remonter jusqu'à la racine
    i = len(src_ancestors) - 1
    j = len(dest_ancestors) - 1
    while i >= 0 and j >= 0 and src_ancestors[i][0] == dest_ancestors[j][0]:
        i -= 1
        j -= 1
    # Concaténer les chemins de src et dest jusqu'à l'ancêtre commun
    path =src_ancestors[:i+2]
    path[i+1][1]=0 #Car on ne prend en compte la puissance de l'ancêtre vers son parent
    path.extend(reversed(dest_ancestors[:j+1]))
    power, chemin=max([x[1] for x in path]), [i[0] for i in path]
    return chemin,power


def routes_out(T,x):
    f=open(data_path+"routes."+str(x)+".out","w")  
    with open(data_path+"routes."+str(x)+".in","r") as file:
        n=file.readline()
        f.write(n)
        for _ in range(int(n)):
            city1,city2,utility=file.readline().split()
            p=str(min_power_tree(T,int(city1),int(city2))[1])
            f.write(city1+" "+city2+" "+utility+" "+p+"\n") 
    f.close()

routes_out(T1,1)
routes_out(T2,2)
routes_out(T3,3)
routes_out(T4,4)
routes_out(T5,5)
routes_out(T6,6)
routes_out(T7,7)
routes_out(T8,8)
routes_out(T9,9)
routes_out(T10,10)