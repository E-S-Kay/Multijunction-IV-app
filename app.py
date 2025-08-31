import streamlit as st
import numpy as np
from scipy.optimize import root_scalar
import plotly.graph_objects as go
import pandas as pd

st.title("Tandem-Solarzellen IV-Simulator")

# ---------------------------------
# Eingabeparameter (Text -> float)
# ---------------------------------
def get_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    try:
        return float(val_str)
    except ValueError:
        st.sidebar.error(f"Ungültige Eingabe bei {label}, Standardwert wird genutzt.")
        return float(default)

st.sidebar.header("Parameter Zelle 1")
Jph1 = get_input("Photostrom Jph1 [mA/cm²]", 30)
J01  = get_input("Sättigungsstrom J01 [mA/cm²]", 1e-10)
n1   = get_input("Idealfaktor n1", 1.0)
Rs1  = get_input("Serienwiderstand Rs1 [Ωcm²]", 0.2)
Rsh1 = get_input("Parallelwiderstand Rsh1 [Ωcm²]", 1000)

st.sidebar.header("Parameter Zelle 2")
Jph2 = get_input("Photostrom Jph2 [mA/cm²]", 25)
J02  = get_input("Sättigungsstrom J02 [mA/cm²]", 1e-12)
n2   = get_input("Idealfaktor n2", 1.2)
Rs2  = get_input("Serienwiderstand Rs2 [Ωcm²]", 0.3)
Rsh2 = get_input("Parallelwiderstand Rsh2 [Ωcm²]", 1500)

T = get_input("Temperatur [K]", 298)

# ---------------------------------
# Konstanten
# ---------------------------------
q = 1.602e-19
k = 1.381e-23
Vth = k*T/q

# ---------------------------------
# Eindiodengleichung (mA/cm²)
# ---------------------------------
def J_diode(V, Jph, J0, n, Rs, Rsh):
    # Funktion, deren Nullstelle Spannung liefert
    def f(Vguess, Jtarget):
        return (Jph 
                - J0*(np.exp((Vguess + Jtarget*Rs)/(n*Vth)) - 1) 
                - (Vguess + Jtarget*Rs)/Rsh 
                - Jtarget)
    return f

# ---------------------------------
# Voc bestimmen
# ---------------------------------
def calc_Voc(Jph, J0, n, Rs, Rsh):
    f = lambda V: (Jph 
                   - J0*(np.exp((V)/(n*Vth)) - 1) 
                   - (V)/Rsh)
    try:
        sol = root_scalar(f, bracket=[0, 2], method="bisect")
        return sol.root
    except:
        return np.nan

# ---------------------------------
# IV-Kennlinie berechnen
# ---------------------------------
def solve_IV(Jph, J0, n, Rs, Rsh, J_common):
    Vsol = []
    for J in J_common:
        f = J_diode(0, Jph, J0, n, Rs, Rsh)
        try:
            sol = root_scalar(lambda V: f(V, J), bracket=[-0.5, 2], method="bisect")
            Vsol.append(sol.root)
        except:
            Vsol.append(np.nan)
    return np.array(Vsol)

# Gemeinsame Stromachse
J_common = np.linspace(0, max(Jph1, Jph2), 400)

# Teilzellen
Voc1 = calc_Voc(Jph1, J01, n1, Rs1, Rsh1)
V1 = solve_IV(Jph1, J01, n1, Rs1, Rsh1, J_common)

Voc2 = calc_Voc(Jph2, J02, n2, Rs2, Rsh2)
V2 = solve_IV(Jph2, J02, n2, Rs2, Rsh2, J_common)

# Tandemspannung = Summe
V_tandem = V1 + V2
Voc_tandem = Voc1 + Voc2

# ---------------------------------
# MPP finden
# ---------------------------------
P_tandem = V_tandem * J_common
idx_mpp = np.nanargmax(P_tandem)
V_mpp, J_mpp, P_mpp = V_tandem[idx_mpp], J_common[idx_mpp], P_tandem[idx_mpp]

P1 = V1 * J_common
idx1 = np.nanargmax(P1)
V1_mpp, J1_mpp, P1_mpp = V1[idx1], J_common[idx1], P1[idx1]

P2 = V2 * J_common
idx2 = np.nanargmax(P2)
V2_mpp, J2_mpp, P2_mpp = V2[idx2], J_common[idx2], P2[idx2]

# ---------------------------------
# IV-Plot
# ---------------------------------
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
    hovermode="x unified",
    xaxis=dict(range=[-0.2, Voc_tandem+0.1])
)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------
# Tabelle der Kenngrößen
# ---------------------------------
def calculate_params(Jsc, Voc, V_mpp, J_mpp, P_mpp):
    FF = (V_mpp * J_mpp) / (Voc * Jsc) if Voc > 0 and Jsc > 0 else np.nan
    Pin = 100.0  # mW/cm²
    PCE = (V_mpp * J_mpp) / Pin * 100  # %
    return {"Jsc [mA/cm²]": Jsc, "Voc [V]": Voc,
            "Vmpp [V]": V_mpp, "Jmpp [mA/cm²]": J_mpp,
            "FF": FF, "PCE [%]": PCE}

params1 = calculate_params(max(J_common), Voc1, V1_mpp, J1_mpp, P1_mpp)
params2 = calculate_params(max(J_common), Voc2, V2_mpp, J2_mpp, P2_mpp)
paramsT = calculate_params(max(J_common), Voc_tandem, V_mpp, J_mpp, P_mpp)

df = pd.DataFrame([params1, params2, paramsT],
                  index=["Zelle 1", "Zelle 2", "Tandem"])

st.subheader("Kenngrößen der Solarzellen")
st.dataframe(df.style.format({
    "Jsc [mA/cm²]": "{:.2f}",
    "Voc [V]": "{:.3f}",
    "Vmpp [V]": "{:.3f}",
    "Jmpp [mA/cm²]": "{:.2f}",
    "FF": "{:.3f}",
    "PCE [%]": "{:.2f}"
}))
