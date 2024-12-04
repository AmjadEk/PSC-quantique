Ceci est le code de notre PSC.
Pour l'instant il y a le code pour l'algorithme quantique :
**ce qui a été fait : **
- définition de l'hamiltonien d'Anderson avec la possibilité de choisir le nombre d'impuretés
- implémentation du HVA (annealing) mais qui reste à corriger
- circuits permettant le calcul d'exponentielle de chaînes de Pauli
- circuit qui calcul la fonction $e^{i H t} c_{i, \sigma}(t) e^{- i H t}c_{j, \sigma'}$
- implémentation du mapping des opérateurs de création et d'annihilation en qubits via Jordan-Wigner
- Test de hadamard (partie réelle et imaginaire) dont il reste à fabriquer une porte Controlled - $e^{i H t} c_{i, \sigma}(t) e^{- i H t}c_{j, \sigma'}$ à partir du circuit permettant de la calculer
