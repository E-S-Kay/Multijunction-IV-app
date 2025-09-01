import numpy as np
import streamlit as st
from scipy.optimize import root_scalar, fsolve
from scipy import stats
import plotly.graph_objects as go

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

def calculate_iv(Jph_mA, J0_mA, n, Rs, Rsh, T, J_common):
    # Umrechnung in A/cm²
    Jph = Jph_mA / 1000.0
    J0  = J0_mA  / 1000.0

    cell = {"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T}
    Voc = estimate_Voc(cell)

    V_vals = np.zeros_like(J_common)
    V_prev = Voc

    for i, JmA in enumerate(J_common):
        J = JmA / 1000.0  # A/cm²
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

    P_plot = V_vals * J_common
    idx_mpp = int(np.nanargmax(P_plot))

    # Berechne Jsc mit diode_equation bei V=0
    try:
        sol = root_scalar(lambda J: diode_equation_V(0.0, J/1000.0, cell),
                          bracket=[0, Jph_mA+5], method="bisect")
        Jsc = sol.root
    except Exception:
        Jsc = Jph_mA

    return V_vals, P_plot, Voc, V_vals[idx_mpp], J_common[idx_mpp], P_plot[idx_mpp], Jsc

# Lineare Interpolation (2 Punkte) für Tandem Jsc
def interpolate_Jsc_two_points(V, J):
    if np.all(V > 0) or np.all(V < 0):
        return np.nan
    idx_pos = np.argmax(V > 0)
    idx_neg = idx_pos - 1
    V_pair = np.array([V[idx_neg], V[idx_pos]])
    J_pair = np.array([J[idx_neg], J[idx_pos]])
    slope, intercept, _, _, _ = stats.linregress(V_pair, J_pair)
    return intercept

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IV-Kennlinie einer Tandemsolarzelle (2 Teilzellen, Eindiodenmodell)")

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
# Berechnung Tandem
# -----------------------------
J_common = np.linspace(0, max(Jph1, Jph2), 400)

# Teilzellen-Spannungen bei gleichem J
V1, P1, Voc1, V1_mpp, J1_mpp, P1_mpp, Jsc1 = calculate_iv(Jph1, J01, n1, Rs1, Rsh1, T1, J_common)
V2, P2, Voc2, V2_mpp, J2_mpp, P2_mpp, Jsc2 = calculate_iv(Jph2, J02, n2, Rs2, Rsh2, T2, J_common)

# Tandem-Kombination
V_tandem = V1 + V2
P_tandem = V_tandem * J_common

idx_mpp_t = int(np.nanargmax(P_tandem))
Voc_tandem = V_tandem[0]  # bei J=0
V_mpp = V_tandem[idx_mpp_t]
J_mpp = J_common[idx_mpp_t]
P_mpp = P_tandem[idx_mpp_t]

# Tandem Jsc mit 2-Punkte-Interpolation
Jsc_tandem = interpolate_Jsc_two_points(V_tandem, J_common)

# -----------------------------
# Ergebnisse in Tabelle
# -----------------------------
def calc_FF(Jsc, Voc, Jmpp, Vmpp):
    if Jsc == 0 or Voc == 0:
        return 0
    return (Jmpp * Vmpp) / (Jsc * Voc)

def calc_PCE(Jsc, Voc, FF):
    return Jsc * Voc * FF / 100.0  # in % (mW/cm² bei 100 mW/cm² Einstrahlung)

FF1 = calc_FF(Jsc1, Voc1, J1_mpp, V1_mpp)
PCE1 = calc_PCE(Jsc1, Voc1, FF1)

FF2 = calc_FF(Jsc2, Voc2, J2_mpp, V2_mpp)
PCE2 = calc_PCE(Jsc2, Voc2, FF2)

FF_t = calc_FF(Jsc_tandem, Voc_tandem, J_mpp, V_mpp)
PCE_t = calc_PCE(Jsc_tandem, Voc_tandem, FF_t)

import pandas as pd
results = pd.DataFrame({
    "Zelle": ["Zelle 1", "Zelle 2", "Tandem"],
    "Jsc [mA/cm²]": [f"{Jsc1:.2f}", f"{Jsc2:.2f}", f"{Jsc_tandem:.2f}"],
    "Voc [V]": [f"{Voc1:.2f}", f"{Voc2:.2f}", f"{Voc_tandem:.2f}"],
    "FF": [f"{FF1:.2f}", f"{FF2:.2f}", f"{FF_t:.2f}"],
    "PCE [%]": [f"{PCE1:.2f}", f"{PCE2:.2f}", f"{PCE_t:.2f}"],
    "Jmpp [mA/cm²]": [f"{J1_mpp:.2f}", f"{J2_mpp:.2f}", f"{J_mpp:.2f}"],
    "Vmpp [V]": [f"{V1_mpp:.2f}", f"{V2_mpp:.2f}", f"{V_mpp:.2f}"]
})

st.table(results)

# -----------------------------
# Interaktive Plots (Plotly)
# -----------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=V1, y=J_common, mode="lines", name="Zelle 1"))
fig1.add_trace(go.Scatter(x=V2, y=J_common, mode="lines", name="Zelle 2"))
fig1.add_trace(go.Scatter(x=V_tandem, y=J_common, mode="lines", name="Tandem", line=dict(width=3)))
fig1.add_trace(go.Scatter(x=[V_mpp], y=[J_mpp], mode="markers", name="Tandem MPP",
                          marker=dict(color="red", size=10, symbol="x")))
fig1.update_layout(
    title="IV-Kennlinien",
    xaxis_title="Spannung [V]",
    yaxis_title="Stromdichte [mA/cm²]",
    xaxis=dict(range=[-0.2, Voc_tandem + 0.1]),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)
