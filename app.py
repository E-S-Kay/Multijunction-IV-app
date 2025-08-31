import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
import pandas as pd

st.title("Tandem-Solarzellen IV-Simulator")

# -----------------------------
# Hilfsfunktion: Umwandeln von Eingaben
# -----------------------------
def to_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default

# -----------------------------
# Eingaben Zelle 1
# -----------------------------
st.sidebar.header("Parameter Zelle 1")
Jph1_str = st.sidebar.text_input("Photostromdichte Jph [mA/cm²] (Z1)", "30")
J01_str  = st.sidebar.text_input("Sättigungsstromdichte J0 [mA/cm²] (Z1)", "1e-10")
n1_str   = st.sidebar.text_input("Idealfaktor n (Z1)", "1")
Rs1_str  = st.sidebar.text_input("Serienwiderstand Rs [Ω·cm²] (Z1)", "0.2")
Rsh1_str = st.sidebar.text_input("Shunt-Widerstand Rsh [Ω·cm²] (Z1)", "1000")
T1_str   = st.sidebar.text_input("Temperatur [K] (Z1)", "298")

# -----------------------------
# Eingaben Zelle 2
# -----------------------------
st.sidebar.header("Parameter Zelle 2")
Jph2_str = st.sidebar.text_input("Photostromdichte Jph [mA/cm²] (Z2)", "18")
J02_str  = st.sidebar.text_input("Sättigungsstromdichte J0 [mA/cm²] (Z2)", "1e-12")
n2_str   = st.sidebar.text_input("Idealfaktor n (Z2)", "1.2")
Rs2_str  = st.sidebar.text_input("Serienwiderstand Rs [Ω·cm²] (Z2)", "0.3")
Rsh2_str = st.sidebar.text_input("Shunt-Widerstand Rsh [Ω·cm²] (Z2)", "1500")
T2_str   = st.sidebar.text_input("Temperatur [K] (Z2)", "298")

# -----------------------------
# Umwandeln in floats
# -----------------------------
Jph1, J01, n1, Rs1, Rsh1, T1 = [to_float(x) for x in [Jph1_str, J01_str, n1_str, Rs1_str, Rsh1_str, T1_str]]
Jph2, J02, n2, Rs2, Rsh2, T2 = [to_float(x) for x in [Jph2_str, J02_str, n2_str, Rs2_str, Rsh2_str, T2_str]]

# -----------------------------
# Konstanten
# -----------------------------
q = 1.602e-19  # C
k = 1.381e-23  # J/K

# -----------------------------
# Eindiodenmodell: Berechnung V(J)
# -----------------------------
def diode_equation(V, J, Jph, J0, n, Rs, Rsh, T):
    return J - (Jph - J0 * (np.exp(q * (V + J*Rs) / (n*k*T)) - 1) - (V + J*Rs)/Rsh)

def get_IV_curve(J_axis, Jph, J0, n, Rs, Rsh, T):
    V = []
    for J in J_axis:  # A/cm²
        func = lambda V: diode_equation(V, J, Jph*1e-3, J0, n, Rs, Rsh, T)
        V_sol = fsolve(func, 0)[0]
        V.append(V_sol)
    return np.array(V)

# -----------------------------
# Gemeinsame Stromachse
# -----------------------------
J_axis = np.linspace(0, min(Jph1, Jph2), 400) * 1e-3  # A/cm²
J_mA = J_axis * 1e3  # mA/cm² für Plots

# IV-Kennlinien
V1 = get_IV_curve(J_axis, Jph1, J01, n1, Rs1, Rsh1, T1)
V2 = get_IV_curve(J_axis, Jph2, J02, n2, Rs2, Rsh2, T2)
V_tandem = V1 + V2

# -----------------------------
# Voc-Berechnung (Nullstrom)
# -----------------------------
def find_Voc(Jph, J0, n, Rs, Rsh, T):
    func = lambda V: Jph*1e-3 - J0*(np.exp(q*V/(n*k*T))-1) - V/Rsh
    return fsolve(func, 0.7)[0]

Voc1 = find_Voc(Jph1, J01, n1, Rs1, Rsh1, T1)
Voc2 = find_Voc(Jph2, J02, n2, Rs2, Rsh2, T2)
Voc_tandem = Voc1 + Voc2

# -----------------------------
# MPP suchen
# -----------------------------
P_tandem = V_tandem * J_axis  # W/cm²
idx_mpp = np.argmax(P_tandem)
V_mpp, J_mpp, P_mpp = V_tandem[idx_mpp], J_mA[idx_mpp], P_tandem[idx_mpp]

# -----------------------------
# Kenngrößen berechnen
# -----------------------------
def calculate_params(Jph, Voc, V, J, V_mpp, J_mpp):
    Jsc = Jph
    FF = (V_mpp * J_mpp*1e-3) / (Voc * Jsc*1e-3) if Voc>0 and Jsc>0 else np.nan
    Pin = 100.0  # mW/cm²
    PCE = (V_mpp * J_mpp) / Pin * 100  # %
    return {"Jsc [mA/cm²]": Jsc, "Voc [V]": Voc,
            "Vmpp [V]": V_mpp, "Jmpp [mA/cm²]": J_mpp,
            "FF": FF, "PCE [%]": PCE}

params1 = calculate_params(Jph1, Voc1, V1, J_mA, V1[idx_mpp], J_mA[idx_mpp])
params2 = calculate_params(Jph2, Voc2, V2, J_mA, V2[idx_mpp], J_mA[idx_mpp])
paramsT = calculate_params(min(Jph1, Jph2), Voc_tandem, V_tandem, J_mA, V_mpp, J_mpp)

df = pd.DataFrame([params1, params2, paramsT],
                  index=["Zelle 1", "Zelle 2", "Tandem"])

# -----------------------------
# Plot mit Plotly
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=V1, y=J_mA, name="Zelle 1"))
fig.add_trace(go.Scatter(x=V2, y=J_mA, name="Zelle 2"))
fig.add_trace(go.Scatter(x=V_tandem, y=J_mA, name="Tandem", line=dict(width=3)))

fig.update_layout(
    xaxis_title="Spannung [V]",
    yaxis_title="Stromdichte [mA/cm²]",
    title="IV-Kennlinien",
)

# Achsen dynamisch verschiebbar
fig.update_xaxes(rangeslider_visible=False)
fig.update_yaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tabelle anzeigen
# -----------------------------
st.subheader("Kenngrößen der Solarzellen")
st.dataframe(df.style.format({
    "Jsc [mA/cm²]": "{:.2f}",
    "Voc [V]": "{:.3f}",
    "Vmpp [V]": "{:.3f}",
    "Jmpp [mA/cm²]": "{:.2f}",
    "FF": "{:.3f}",
    "PCE [%]": "{:.2f}"
}))
