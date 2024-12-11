import numpy as np



def tridiag(diag,sousdiag,surdiag):
    """Construit une matrice tridiagonale à partir des
    listes des éléments des trois diagonales"""
    n = len(diag)
    A = np.zeros((n,n))
    A[0][0] = diag[0]
    A[0][1] = surdiag[0]
    for i in np.arange(1,n-1):
        A[i][i] = diag[i]
        A[i][i-1] = sousdiag[i-1]
        A[i][i+1] = surdiag[i]
    A[n-1][n-2] = sousdiag[n-2]
    A[n-1][n-1] = diag[n-1]
    return A



def lanczos(v,H,M,eps):
    """Calcule les valeurs propres approchées de la Matrice H avec l'algorithme de Lanczos.
    v : vecteur initial (choisir un état de Fock ?)
    M : nombre d'itérations
    eps : seuil de précision"""
    a = []
    b = []
    b.append(np.linalg.norm(v))
    v = v/b[-1]
    w = np.zeros_like(v)
    w = w + np.matmul(H,v)
    a.append(np.dot(v,w))
    w = w -a[-1] * v
    b.append(np.linalg.norm(w))
    T = []
    for i in range(M):
        if np.abs(b[-1])< eps:
            break
        w = 1/b[-1] * w
        v = - b[-1] * v
        v, w = w, v
        w = w + np.matmul(H,v)
        a.append(np.dot(v,w))
        w = w -a[-1] * v
        b.append(np.linalg.norm(w))
    b = b[1:-1] 
    (a,b) = (np.array(a),np.array(b))
    A = np.diag(a) + np.diag(b, k = -1) + np.diag(b, k =1)
    return np.linalg.eigvals(A)


n = 1000
m = 100


H = np.zeros((n,n))

for k in range(m):
    i = np.random.randint(1,n)
    j = np.random.randint(1,n)
    v = np.random.randint(1,n)
    H[i][j] = v

H = H + H.T



v = np.ones(n)
M = n-1
epsilon = 10** (-10)


Lambda_vraie = np.sort(lanczos(v,H,M,epsilon))[0]



Lambda = np.sort(np.real(np.linalg.eigvals(H)))[0]



print(Lambda_vraie)
print(Lambda)
print(Lambda-Lambda_vraie)
















