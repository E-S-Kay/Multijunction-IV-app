import streamlit as st
import numpy as np
from scipy.optimize import root_scalar
import plotly.graph_objects as go
import pandas as pd

# Konstanten
q = 1.602e-19  # Elementarladung [C]
k_B = 1.381e-23  # Boltzmann-Konstante [J/K]

st.title("Tandem-Solarzelle IV-Kennlinie")

# -----------------------------
# Eingaben für beide Zellen
# -----------------------------
st.sidebar.header("Parameter Zelle 1")
Jph1_str = st.sidebar.text_input("Photostromdichte Jph [mA/cm²] (Z1)", "30")
J01_str = st.sidebar.text_input("Sättigungsstromdichte J0 [mA/cm²] (Z1)", "1e-10")
n1_str = st.sidebar.text_input("Idealfaktor n (Z1)", "1")
Rs1_str = st.sidebar.text_input("Serienwiderstand Rs [Ω·cm²] (Z1)", "0.2")
Rsh1_str = st.sidebar.text_input("Shunt-Widerstand Rsh [Ω·cm²] (Z1)", "1000")
T1_str = st.sidebar.text_input("Temperatur [K] (Z1)", "298")

st.sidebar.header("Parameter Zelle 2")
Jph2_str = st.sidebar.text_input("Photostromdichte Jph [mA/cm²] (Z2)", "18")
J02_str = st.sidebar.text_input("Sättigungsstromdichte J0 [mA/cm²] (Z2)", "1e-12")
n2_str = st.sidebar.text_input("Idealfaktor n (Z2)", "1.2")
Rs2_str = st.sidebar.text_input("Serienwiderstand Rs [Ω·cm²] (Z2)", "0.3")
Rsh2_str = st.sidebar.text_input("Shunt-Widerstand Rsh [Ω·cm²] (Z2)", "1500")
T2_str = st.sidebar.text_input("Temperatur [K] (Z2)", "298")


def to_float(val, default):
    try:
        return float(val)
    except ValueError:
        return default

# Umwandlung Eingaben
Jph1 = to_float(Jph1_str, 30) * 1e-3
J01 = to_float(J01_str, 1e-10) * 1e-3
n1 = to_float(n1_str, 1)
Rs1 = to_float(Rs1_str, 0.2)
Rsh1 = to_float(Rsh1_str, 1000)
T1 = to_float(T1_str, 298)

Jph2 = to_float(Jph2_str, 18) * 1e-3
J02 = to_float(J02_str, 1e-12) * 1e-3
n2 = to_float(n2_str, 1.2)
Rs2 = to_float(Rs2_str, 0.3)
Rsh2 = to_float(Rsh2_str, 1500)
T2 = to_float(T2_str, 298)

# -----------------------------
# Hilfsfunktionen
# -----------------------------
def solve_voltage(J, Jph, J0, n, Rs, Rsh, T):
    Vt = n * k_B * T / q
    def f(V):
        return Jph - J0*(np.exp((V+J*Rs)/Vt)-1) - (V+J*Rs)/Rsh - J
    try:
        sol = root_scalar(f, bracket=[-2, 3], method="bisect", xtol=1e-6, maxiter=100)
        if sol.converged:
            return sol.root
        else:
            return np.nan
    except ValueError:
        return np.nan

def get_IV_curve(Jph, J0, n, Rs, Rsh, T, J_common):
    V = []
    for J in J_common:
        V.append(solve_voltage(J, Jph, J0, n, Rs, Rsh, T))
    return np.array(V)

def find_Voc(Jph, J0, n, Rs, Rsh, T):
    Vt = n * k_B * T / q
    def f(V):
        return Jph - J0*(np.exp((V)/Vt)-1) - V/Rsh
    try:
        sol = root_scalar(f, bracket=[0, 2], method="bisect", xtol=1e-6, maxiter=100)
        if sol.converged:
            return sol.root
        else:
            return np.nan
    except ValueError:
        return np.nan

def calculate_params(V, J, Voc):
    Jsc = max(J)  # [mA/cm²]
    P = V * J
    idx = np.nanargmax(P)
    V_mpp = V[idx]
    J_mpp = J[idx]
    P_mpp = P[idx]
    FF = (V_mpp * J_mpp) / (Voc * Jsc) if Voc > 0 and Jsc > 0 else np.nan
    Pin = 100.0  # mW/cm²
    PCE = (V_mpp * J_mpp) / Pin * 100
    return {"Jsc [mA/cm²]": Jsc, "Voc [V]": Voc,
            "Vmpp [V]": V_mpp, "Jmpp [mA/cm²]": J_mpp,
            "FF": FF, "PCE [%]": PCE}

# -----------------------------
# Berechnung Kennlinien
# -----------------------------
Jsc_est = min(Jph1, Jph2) * 1e3
J_common = np.linspace(0, Jsc_est, 300)  # [mA/cm²]

J_common_A = J_common * 1e-3  # [A/cm²]

V1 = get_IV_curve(J_common_A, Jph1, J01, n1, Rs1, Rsh1, T1)
V2 = get_IV_curve(J_common_A, Jph2, J02, n2, Rs2, Rsh2, T2)
V_tandem = V1 + V2

Voc1 = find_Voc(Jph1, J01, n1, Rs1, Rsh1, T1)
Voc2 = find_Voc(Jph2, J02, n2, Rs2, Rsh2, T2)
Voc_tandem = Voc1 + Voc2

# -----------------------------
# Tabelle mit Kenngrößen
# -----------------------------
params1 = calculate_params(V1, J_common, Voc1)
params2 = calculate_params(V2, J_common, Voc2)
paramsT = calculate_params(V_tandem, J_common, Voc_tandem)

df = pd.DataFrame([params1, params2, paramsT],
                  index=["Zelle 1", "Zelle 2", "Tandem"])

# -----------------------------
# Plot
# -----------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=V1, y=J_common, mode="lines", name="Zelle 1"))
fig1.add_trace(go.Scatter(x=V2, y=J_common, mode="lines", name="Zelle 2"))
fig1.add_trace(go.Scatter(x=V_tandem, y=J_common, mode="lines", name="Tandem", line=dict(width=3)))
fig1.update_layout(
    title="IV-Kennlinien",
    xaxis_title="Spannung [V]",
    yaxis_title="Stromdichte [mA/cm²]",
    hovermode="x unified",
    xaxis=dict(range=[-0.2, Voc_tandem+0.1])
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Kenngrößen der Solarzellen")
st.dataframe(df.style.format({
    "Jsc [mA/cm²]": "{:.2f}",
    "Voc [V]": "{:.3f}",
    "Vmpp [V]": "{:.3f}",
    "Jmpp [mA/cm²]": "{:.2f}",
    "FF": "{:.3f}",
    "PCE [%]": "{:.2f}"
}))
