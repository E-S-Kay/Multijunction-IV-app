import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import root_scalar, fsolve

# -----------------------------
# Hilfsfunktionen
# -----------------------------
def safe_exp(x):
    return np.exp(np.clip(x, -700, 700))

def diode_equation_V(V, J, cell):
    q = 1.602176634e-19  # C
    k = 1.380649e-23     # J/K
    arg = q * (V + J * cell["Rs"]) / (cell["n"] * k * cell["T"])
    exp_term = safe_exp(arg)
    return J - (cell["Jph"] - cell["J0"] * (exp_term - 1.0) - (V + J * cell["Rs"]) / cell["Rsh"])

def estimate_Voc(cell):
    try:
        sol = root_scalar(lambda V: diode_equation_V(V, 0.0, cell),
                          bracket=[-0.5, 2.0], method="bisect")
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return 0.6

def calculate_iv(Jph_mA, J0_mA, n, Rs, Rsh, T, Npts=400):
    # Umrechnung in A/cm²
    Jph = Jph_mA / 1000.0
    J0  = J0_mA  / 1000.0

    cell = {"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T}
    Voc = estimate_Voc(cell)

    J_vals = np.linspace(0.0, Jph, Npts)
    V_vals = np.zeros_like(J_vals)
    V_prev = Voc

    for i, J in enumerate(J_vals):
        V_sol = None
        try:
            sol = root_scalar(lambda V: diode_equation_V(V, J, cell),
                              bracket=[-1.0, Voc+1.5], method="bisect")
            if sol.converged:
                V_sol = sol.root
        except Exception:
            pass
        if V_sol is None:
            guess = V_prev
            try:
                sol = fsolve(lambda V: diode_equation_V(V, J, cell), guess)
                V_sol = sol[0]
            except Exception:
                V_sol = guess
        V_vals[i] = V_sol
        V_prev = V_sol

    J_plot = J_vals * 1000.0   # zurück in mA/cm²
    P_plot = V_vals * J_plot   # Leistung

    idx_mpp = int(np.nanargmax(P_plot))
    return V_vals, J_plot, P_plot, Voc, V_vals[idx_mpp], J_plot[idx_mpp], P_plot[idx_mpp]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IV-Kennlinie einer Tandemsolarzelle (2 Teilzellen)")

st.sidebar.header("Parameter der ersten Zelle")
def get_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    try:
        return float(val_str)
    except ValueError:
        st.sidebar.error(f"Ungültige Eingabe für {label}.")
        return float(default)

# Eingaben Zelle 1
Jph1 = get_input("Zelle 1: Photostrom Jph [mA/cm²]", 30.0)
J01  = get_input("Zelle 1: Sättigungsstrom J0 [mA/cm²]", 1e-10)
n1   = get_input("Zelle 1: Idealfaktor n", 1.0)
Rs1  = get_input("Zelle 1: Serienwiderstand Rs [Ohm·cm²]", 0.2)
Rsh1 = get_input("Zelle 1: Parallelwiderstand Rsh [Ohm·cm²]", 1000.0)
T1   = get_input("Zelle 1: Temperatur T [K]", 298.0)

st.sidebar.header("Parameter der zweiten Zelle")
# Eingaben Zelle 2
Jph2 = get_input("Zelle 2: Photostrom Jph [mA/cm²]", 20.0)
J02  = get_input("Zelle 2: Sättigungsstrom J0 [mA/cm²]", 1e-12)
n2   = get_input("Zelle 2: Idealfaktor n", 1.0)
Rs2  = get_input("Zelle 2: Serienwiderstand Rs [Ohm·cm²]", 0.2)
Rsh2 = get_input("Zelle 2: Parallelwiderstand Rsh [Ohm·cm²]", 1000.0)
T2   = get_input("Zelle 2: Temperatur T [K]", 298.0)

# -----------------------------
# Berechnung Teilzellen
# -----------------------------
V1, J1, P1, Voc1, V1_mpp, J1_mpp, P1_mpp = calculate_iv(Jph1, J01, n1, Rs1, Rsh1, T1)
V2, J2, P2, Voc2, V2_mpp, J2_mpp, P2_mpp = calculate_iv(Jph2, J02, n2, Rs2, Rsh2, T2)

# -----------------------------
# Berechnung Tandemzelle
# -----------------------------
# Der Strom ist durch die schwächere Zelle limitiert
J_common = np.linspace(0, min(J1.max(), J2.max()), 400)

# Interpolation der Spannungen bei gegebenem Strom
V1_interp = np.interp(J_common, J1[::-1], V1[::-1])
V2_interp = np.interp(J_common, J2[::-1], V2[::-1])

V_tandem = V1_interp + V2_interp
P_tandem = V_tandem * J_common

idx_mpp_t = int(np.nanargmax(P_tandem))
Voc_tandem = V_tandem[0]  # bei J=0

# -----------------------------
# Ergebnisse anzeigen
# -----------------------------
st.write(f"**Zelle 1 Voc** = {Voc1:.4f} V, **Zelle 2 Voc** = {Voc2:.4f} V")
st.write(f"**Tandem Voc** = {Voc_tandem:.4f} V")
st.write(f"**Tandem MPP**: V = {V_tandem[idx_mpp_t]:.4f} V, J = {J_common[idx_mpp_t]:.4f} mA/cm², P = {P_tandem[idx_mpp_t]:.4f} mW/cm²")

# -----------------------------
# Plots
# -----------------------------
# IV-Kurven
fig1, ax1 = plt.subplots()
ax1.plot(V1, J1, label="Zelle 1")
ax1.plot(V2, J2, label="Zelle 2")
ax1.plot(V_tandem, J_common, label="Tandem", linewidth=2)
ax1.scatter([V_tandem[idx_mpp_t]], [J_common[idx_mpp_t]], color="red", label="Tandem MPP")
ax1.set_xlabel("Spannung [V]")
ax1.set_ylabel("Stromdichte [mA/cm²]")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# P-V-Kurve Tandem
fig2, ax2 = plt.subplots()
ax2.plot(V_tandem, P_tandem, label="Tandem P-V")
ax2.scatter([V_tandem[idx_mpp_t]], [P_tandem[idx_mpp_t]], color="red", label="MPP")
ax2.set_xlabel("Spannung [V]")
ax2.set_ylabel("Leistung [mW/cm²]")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
