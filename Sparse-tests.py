import numpy as np 
import scipy.sparse as spr
import time
import matplotlib.pyplot as plt

def matrice_creuse_sym(n,m):
    H = np.zeros((n,n))
    for k in range(m):
        i = np.random.randint(1,n)
        j = np.random.randint(1,n)
        v = np.random.randint(1,n)
        H[i][j] = v
    H = H + H.T
    return H


def test_temps(n,sparcity):
    m = int(n * sparcity)
    H = matrice_creuse_sym(n,m)
    v = np.ones(n)
    H_sparse = spr.csr_array(H)

    #test pour le produit matriciel classique

    st = time.time()
    w = H@v
    et = time.time()
    elapsed_t = et - st

    #test pour le produit matriciel compressé

    st = time.time()
    w2 = H_sparse@v
    et = time.time()
    elapsed_t_spr = et - st


    return (elapsed_t, elapsed_t_spr)



def graphes_efficacite(N,Sparsities):
    for spar in Sparsities:
        T1 = []
        T2 = []
        for n in  N:
            t1,t2 = test_temps(n,spar)
            T1.append(t1)
            T2.append(t2)
        plt.figure()
        plt.xlabel(r"$n$")
        plt.xscale("log")
        plt.ylabel(r"Temps ($s$)")
        plt.scatter(N, T1, color = 'b', label = "Calcul matriciel naïf")
        plt.title("Temps avec une sparsité de " + str(spar))
        plt.legend()

        plt.figure()
        plt.xlabel(r"$n$")
        plt.xscale("log")
        plt.ylabel(r"Temps ($s$)")
        plt.scatter(N, T2, color = 'r', label = "Calcul matriciel optimisé")
        plt.title("Temps avec une sparsité de " + str(spar))
        plt.legend()


        print("Temps naïfs, sparsity = " + str(spar))
        print(T1)
        print("Temps optimisés :")
        print(T2)
        print("\n")
    plt.show()
    return

N = np.linspace(1000,10000,9, dtype = int)
Sparsities = [0.001,0.0001]

graphes_efficacite(N,Sparsities)
        
    
