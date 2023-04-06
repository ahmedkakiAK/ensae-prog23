import numpy as np

from matplotlib import rc

rc('animation', html='jshtml')
import matplotlib
import matplotlib.pyplot as plt
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

fig = plt.figure(figsize=(8, 6))
ax = plt.axes()

from random import *

seed(0)  # Pour assurer la reproductibilité


def cambrioleurs_cst(n, gamma, A0, B0, teta, omega, eta,
                     delta_t):  # matrice du nb de cambrioleurs présents sur chaque site
    lattice = np.zeros((n, n))  # matrice avec tous les cambrioleurs
    n0 = int(gamma * delta_t / (1 - np.exp(-(A0 + B0) * delta_t)))
    lattice += n0
    return lattice


def attractiveness(n, gamma, A0, teta, omega):  # Initialisation de la matrice des attractivités
    A = np.zeros((n, n))  # Matrice des attractivités
    B0 = teta * gamma / omega
    A += A0 + B0
    return A


def neighboring_sites(n, i, j):  # Une fonction qui retourne la liste des indices des sites voisins du site (i,j)
    l = []
    for (a, b) in [(i, j - 1), (i, j + 1), (i + 1, j), (i - 1, j)]:
        if a in range(n) and b in range(n):
            l.append((a, b))
    return l


def criminal_loop(t, lattice, A, delta_t, eta, gamma, teta, omega, l, A0,
                  E):  # E=nb de cambriolages réalisés entre t et t + delta_t
    n = len(lattice)
    P = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            P[i, j] = 1 - np.exp(-A[i, j] * delta_t)
            burglars = lattice[i, j]
            for d in range(int(burglars)):
                if random() < P[i, j]:
                    lattice[i, j] -= 1
                    E[i, j] += 1
                else:
                    somme = 0
                    index = None
                    x = random()
                    inf = 0
                    liste = []
                    for a, b in neighboring_sites(n, i, j):
                        somme += A[a, b]
                    for a, b in neighboring_sites(n, i, j):
                        q = A[a, b] / somme
                        liste.append(q)
                        sup = sum(liste)
                        inf = sup - q
                        if x > inf and x < sup:
                            index = a, b
                    lattice[i, j] -= 1
                    lattice[a, b] += 1

    B = A - A0
    for i in range(n):
        for j in range(n):
            z = len(neighboring_sites(n, i, j))
            B[i, j] = (B[i, j] + ((eta * (l ** 2)) / z) * laplacian(B, l, i, j)) * (1 - omega * delta_t) + teta * E[
                i, j]
            new_burglars = np.random.poisson(gamma)  ##
            lattice[i, j] += new_burglars

    return A0 + B, lattice


def laplacian(B, l, i, j):
    n = len(B)
    sum = 0
    z = len(neighboring_sites(n, i, j))
    for a, b in neighboring_sites(n, i, j):
        sum += B[a, b]
    return (sum - z * B[i, j]) / l ** 2


# fonction modifiée par ChatGPT
def matrice(t, n=10, gamma=0.002, eta=0.03, teta=5.6, delta_t=1 / 100, omega=1 / 15, l=1, A0=1 / 30,
            E=np.zeros((10, 10))):
    B0 = teta * gamma / omega
    lattice = cambrioleurs_cst(n, gamma, A0, B0, teta, omega, eta, delta_t)
    A = attractiveness(n, gamma, A0, teta, omega)
    for i in range(t):
        A, lattice = criminal_loop(i, lattice, A, delta_t, eta, gamma, teta, omega, l, A0, E)

    return A


matrice(1000)
