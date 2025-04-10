import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

################################################################################
#  1) Utility functions for fermionic creation/annihilation in a 4-orbital space
################################################################################

def occupation(state, orbital):
    """
    Returns 1 if 'orbital' is occupied in 'state' (bitmask),
    else 0.  'state' runs 0..15 for 4 orbitals.
    orbital = 0..3
    """
    return (state >> orbital) & 1

def fermion_sign(state, orbital):
    """
    Computes the Jordan-Wigner (-1)^{number_of_occupied_orbitals < orbital}.
    For creation/annihilation of a fermion at 'orbital', we pick up a sign factor
    = (-1)^(sum of occupancies in all orbitals < 'orbital').

    This ensures proper anticommutation for distinct orbitals.
    """
    # Count how many occupied orbitals lie *below* the given orbital index
    count = 0
    for i in range(orbital):
        count += occupation(state, i)
    # sign = (-1)^count
    return +1 if (count % 2) == 0 else -1

def apply_c_dagger(orbital, state):
    """
    Apply c^\dagger_(orbital) to many-body basis state 'state'.
    Returns (new_state, phase) or (None, 0) if it annihilates to 0.
    """
    if occupation(state, orbital) == 1:
        # Already occupied => c^\dagger kills the state
        return None, 0.0
    # Otherwise, we flip that orbital bit from 0->1
    sign = fermion_sign(state, orbital)
    new_state = state ^ (1 << orbital)  # flip the bit
    return new_state, float(sign)

def apply_c(orbital, state):
    """
    Apply c_(orbital) to many-body basis state 'state'.
    Returns (new_state, phase) or (None, 0) if it annihilates to 0.
    """
    if occupation(state, orbital) == 0:
        # It's unoccupied => c kills the state
        return None, 0.0
    # Otherwise, we flip that orbital bit from 1->0
    sign = fermion_sign(state, orbital)
    new_state = state ^ (1 << orbital)
    return new_state, float(sign)

def apply_n(orbital, state):
    """
    Number operator n = c^\dagger c on 'state'.
    Just returns occupancy * state (diagonal in Fock basis).
    """
    return occupation(state, orbital)

################################################################################
#  2) Build the 2-site spinful Anderson Hamiltonian (16x16)
################################################################################

def build_anderson_2site_spinful(e_d, e_b, U_val, V):
    """
    Returns the 16x16 Hamiltonian matrix for:
      H = sum_{sigma} [ e_d n_{d,sigma} + e_b n_{b,sigma}
                        + V(c^\dagger_{d,sigma} c_{b,sigma} + h.c.) ]
          + U * n_{d,up} n_{d,down}

    Orbital ordering (bit indices):
      0 -> d_up
      1 -> d_dn
      2 -> b_up
      3 -> b_dn
    """
    dim = 16
    H = np.zeros((dim, dim), dtype=np.float64)

    # 2a) Diagonal part: onsite energies + interaction
    for s in range(dim):
        nd_up  = apply_n(0, s)  # n_{d, up}
        nd_dn  = apply_n(1, s)  # n_{d, dn}
        nb_up  = apply_n(2, s)  # n_{b, up}
        nb_dn  = apply_n(3, s)  # n_{b, dn}

        # Onsite energies
        E_onsite = e_d*(nd_up + nd_dn) + e_b*(nb_up + nb_dn)
        # Interaction U on impurity if both spins are occupied
        E_int = U_val * nd_up * nd_dn

        H[s, s] = E_onsite + E_int

    # 2b) Off-diagonal part: hopping V (c_d_sigma^\dagger c_b_sigma + h.c.)
    # We'll do this for sigma= up(0->2), dn(1->3).
    pairs = [(0,2), (1,3)]  # (d_up, b_up), (d_dn, b_dn)
    for (d_orb, b_orb) in pairs:
        for s in range(dim):
            # c_d^\dagger c_b on |s>
            # means: first apply c_b to |s>, then apply c_d^\dagger
            s_inter, amp1 = apply_c(b_orb, s)  # c_b |s>
            if s_inter is not None:
                s_final, amp2 = apply_c_dagger(d_orb, s_inter)
                if s_final is not None:
                    H[s_final, s] += V*(amp1*amp2)

            # c_b^\dagger c_d on |s>
            s_inter, amp1 = apply_c(d_orb, s)  # c_d |s>
            if s_inter is not None:
                s_final, amp2 = apply_c_dagger(b_orb, s_inter)
                if s_final is not None:
                    H[s_final, s] += V*(amp1*amp2)

    return H

################################################################################
#  3) Diagonalize, show energies, and optionally build a Green's function
################################################################################

def lanczos_iteration(H, init_vec, max_iter=50, tol=1e-12):
    """
    Standard Lanczos iteration on vector `init_vec` under Hamiltonian `H`.
    Returns alphas, betas, lanczos_vecs.
    """
    n = len(init_vec)
    alphas = []
    betas = []
    lanczos_vecs = []

    # Normalize initial vector
    v0 = init_vec / la.norm(init_vec)
    lanczos_vecs.append(v0)

    w = np.zeros(n, dtype=np.complex128)
    beta = 0.0

    for j in range(max_iter):
        vj = lanczos_vecs[-1]
        w[:] = H @ vj
        alpha = np.vdot(vj, w).real
        w -= alpha * vj
        if j > 0:
            w -= beta * lanczos_vecs[-2]
        alphas.append(alpha)

        beta = la.norm(w)
        betas.append(beta)
        if beta < tol:
            break
        v_new = w / beta
        lanczos_vecs.append(v_new)

    return np.array(alphas), np.array(betas), np.array(lanczos_vecs)

def continued_fraction_green(alphas, betas, omega_vals, eta, norm_f0):
    """
    Compute the Lanczos-based Green's function G(omega) = (1/norm_f0^2) * (0,0) element
    of [omega + i eta - T]^-1, where T is the tridiagonal matrix from (alphas, betas).

    This direct approach inverts the MxM matrix for each omega. For larger M,
    you'd typically use a continued-fraction recursion.
    """
    M = len(alphas)
    Gw = np.zeros_like(omega_vals, dtype=np.complex128)

    # Build tridiagonal T
    T = np.diag(alphas) + np.diag(betas[:-1],1) + np.diag(betas[:-1],-1)

    for i, w in enumerate(omega_vals):
        mat = (w + 1j*eta)*np.eye(M) - T
        inv_mat = la.inv(mat)
        Gw[i] = inv_mat[0,0] / (norm_f0**2)
    return Gw

def main():
    # ---------------------------
    # Model parameters (example):
    # ---------------------------
    e_d  = -1.0  # impurity onsite
    e_b  =  1.0  # bath onsite
    Uval =  2.0  # interaction on impurity
    V    =  0.5  # hopping
    print(f"Parameters: e_d={e_d}, e_b={e_b}, U={Uval}, V={V}")

    # Build the Hamiltonian (16x16)
    H = build_anderson_2site_spinful(e_d, e_b, Uval, V)
    print(H)
    # Diagonalize exactly
    E, Umat = la.eigh(H)  # E ascending
    print("Eigenvalues (energies) from exact diagonalization:")
    for i, val in enumerate(E):
        print(f"  State {i:2d}:  E = {val: .6f}")
    E0 = E[0]
    print(f"Ground-state energy E0 = {E0: .6f}")

    # --------------------------------------------------
    #  Build an operator: c_{d, up}^\dagger (impurity up)
    # --------------------------------------------------
    # We'll use this to illustrate the Green's function A(omega).
    # The operator dimension is also 16x16. We'll fill it similarly.
    dim = 16
    cdag_d_up = np.zeros((dim, dim), dtype=np.float64)
    for s in range(dim):
        s_new, amp = apply_c_dagger(0, s)  # orbital 0 => d_up
        if s_new is not None:
            cdag_d_up[s_new, s] = amp

    #  Construct the initial Krylov vector |f0> = c^\dagger_{d_up} |GS>
    gs = Umat[:,0]  # ground-state wavefunction (column 0)
    f0 = cdag_d_up @ gs
    norm_f0 = la.norm(f0)
    print(f"Norm of c_d_up^\dagger|GS> = {norm_f0: .6e}")

    # Lanczos in that subspace
    alphas, betas, lanc_vecs = lanczos_iteration(H, f0, max_iter=50, tol=1e-14)

    # Evaluate G(omega) in [-4, +4], for example
    omega_vals = np.linspace(-4.0, 4.0, 400)
    eta = 0.05
    # Shift frequencies by E0 => we consider G(omega) = <GS| c (1/(omega - (H-E0)+i eta)) c^\dagger |GS>
    # So effectively we pass (omega - (E-E0)) => (omega_vals - E0) to the T matrix.
    # We'll do the simpler approach: T ~ H - E0 in the Lanczos subspace.
    freq_shifted = omega_vals - E0
    G_omega = continued_fraction_green(alphas, betas, freq_shifted, eta, norm_f0)
    A_omega = -1/np.pi * np.imag(G_omega)

    # Plot the spectral function
    plt.figure(figsize=(6,4))
    plt.plot(omega_vals, A_omega, label="A(omega)")
    plt.axvline(x=E0, color='r', ls='--', label="Ground-state energy E0")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$A(\omega)$")
    plt.title("Spectral Function via Lanczos (2-site spinful Anderson)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
