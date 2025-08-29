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
                          bracket=[-0.5, 1.5], method="bisect")
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return 0.6

def calculate_iv(Jph_mA, J0_mA, n, Rs, Rsh, T, Npts=400):
    # Einheitenumrechnung
    Jph = Jph_mA / 1000.0  # A/cm²
    J0  = J0_mA  / 1000.0  # A/cm²

    cell = {"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T}
    Voc = estimate_Voc(cell)

    J_vals = np.linspace(0.0, Jph, Npts)
    V_vals = np.zeros_like(J_vals)
    V_prev = Voc

    for i, J in enumerate(J_vals):
        V_sol = None
        try:
            sol = root_scalar(lambda V: diode_equation_V(V, J, cell),
                              bracket=[-1.0, Voc+1.0], method="bisect")
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

    # Umwandeln in mA/cm²
    J_plot = J_vals * 1000.0
    P_plot = V_vals * J_plot

    idx_mpp = int(np.nanargmax(P_plot))
    return V_vals, J_plot, P_plot, Voc, V_vals[idx_mpp], J_plot[idx_mpp], P_plot[idx_mpp]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IV-Kennlinie einer Single-Junction Solarzelle")

st.sidebar.header("Eingabeparameter (wissenschaftliche Notation erlaubt, z.B. 1e-10)")

def get_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    try:
        return float(val_str)
    except ValueError:
        st.sidebar.error(f"Ungültige Eingabe für {label}. Bitte Zahl eingeben.")
        return float(default)

# Eingaben (alle als Textfeld)
Jph = get_input("Photostrom Jph [mA/cm²]", 30.0)
J0  = get_input("Sättigungsstrom J0 [mA/cm²]", 1e-10)
n   = get_input("Idealfaktor n", 1.0)
Rs  = get_input("Serienwiderstand Rs [Ohm·cm²]", 0.2)
Rsh = get_input("Parallelwiderstand Rsh [Ohm·cm²]", 1000.0)
T   = get_input("Temperatur T [K]", 298.0)

# Automatische Berechnung sobald Eingaben geändert werden
V, J, P, Voc, V_mpp, J_mpp, P_mpp = calculate_iv(Jph, J0, n, Rs, Rsh, T)

st.write(f"**Leerlaufspannung Voc** = {Voc:.4f} V")
st.write(f"**MPP**: V = {V_mpp:.4f} V, J = {J_mpp:.4f} mA/cm², P = {P_mpp:.4f} mW/cm²")

# Plot IV
fig1, ax1 = plt.subplots()
ax1.plot(V, J, label="IV-Kurve")
ax1.scatter([V_mpp], [J_mpp], color="red", label="MPP")
ax1.set_xlabel("Spannung [V]")
ax1.set_ylabel("Stromdichte [mA/cm²]")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# Plot P-V
fig2, ax2 = plt.subplots()
ax2.plot(V, P, label="P-V-Kurve")
ax2.scatter([V_mpp], [P_mpp], color="red", label="MPP")
ax2.set_xlabel("Spannung [V]")
ax2.set_ylabel("Leistung [mW/cm²]")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)
