import numpy as np
import streamlit as st
from scipy.optimize import root_scalar, fsolve
from scipy import stats
import plotly.graph_objects as go
import pandas as pd

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

    # Jsc (bei V=0 lösen)
    try:
        sol_jsc = root_scalar(lambda J: diode_equation_V(0.0, J/1000.0, cell), bracket=[-Jph_mA*1.2, Jph_mA*1.2])
        Jsc_val = sol_jsc.root if sol_jsc.converged else np.nan
    except Exception:
        Jsc_val = np.nan

    return V_vals, P_plot, Voc, V_vals[idx_mpp], J_common[idx_mpp], P_plot[idx_mpp], Jsc_val

def interpolate_Jsc_two_points(V, J):
    # Finde Index, wo das Vorzeichen wechselt
    sign_changes = np.where(np.diff(np.sign(V)) != 0)[0]
    if len(sign_changes) == 0:
        return np.nan  # kein Schnittpunkt gefunden
    
    idx = sign_changes[0]  # erster Schnittpunkt
    V1, V2 = V[idx], V[idx+1]
    J1, J2 = J[idx], J[idx+1]
    
    # Lineare Interpolation: J(V=0) = J1 + (0 - V1) * (J2 - J1) / (V2 - V1)
    return J1 + (0 - V1) * (J2 - J1) / (V2 - V1)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IV-Kennlinie einer Mehrfachsolarzelle (1–4 Teilzellen, Eindiodenmodell)")

num_cells = st.sidebar.selectbox("Anzahl der Teilzellen", [1, 2, 3, 4], index=1)

cells = []
for i in range(num_cells):
    st.sidebar.header(f"Parameter Zelle {i+1}")
    Jph = float(st.sidebar.text_input(f"Zelle {i+1}: Photostrom Jph [mA/cm²]", "30.0" if i==0 else "20.0"))
    J0  = float(st.sidebar.text_input(f"Zelle {i+1}: Sättigungsstrom J0 [mA/cm²]", "1e-10" if i==0 else "1e-12"))
    n   = float(st.sidebar.text_input(f"Zelle {i+1}: Idealfaktor n", "1.0"))
    Rs  = float(st.sidebar.text_input(f"Zelle {i+1}: Serienwiderstand Rs [Ohm·cm²]", "0.2"))
    Rsh = float(st.sidebar.text_input(f"Zelle {i+1}: Parallelwiderstand Rsh [Ohm·cm²]", "1000.0"))
    T   = float(st.sidebar.text_input(f"Zelle {i+1}: Temperatur T [K]", "298.0"))
    cells.append({"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T})

# -----------------------------
# Berechnung
# -----------------------------
J_common = np.linspace(0, max([c["Jph"] for c in cells]), 800)

V_all, P_all = [], []
results = []

for i, c in enumerate(cells):
    V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv(c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"], J_common)
    V_all.append(V)
    P_all.append(P)
    FF = (Vmpp * Jmpp) / (Voc * Jsc) if (Voc > 0 and Jsc > 0) else np.nan
    PCE = Pmpp / 100.0  # bei 100 mW/cm²
    results.append([f"Zelle {i+1}", f"{Jsc:.2f}", f"{Voc:.2f}", f"{FF*100:.2f}", f"{PCE*100:.2f}", f"{Jmpp:.2f}", f"{Vmpp:.2f}"])

# Tandem-Spannung und Leistung
V_tandem = np.sum(np.vstack(V_all), axis=0)
P_tandem = V_tandem * J_common

idx_mpp_t = int(np.nanargmax(P_tandem))
V_mpp = V_tandem[idx_mpp_t]
J_mpp = J_common[idx_mpp_t]
P_mpp = P_tandem[idx_mpp_t]

Voc_tandem = V_tandem[0]
Jsc_tandem = interpolate_Jsc_tandem(V_tandem, J_common)
FF_tandem = (V_mpp * J_mpp) / (Voc_tandem * Jsc_tandem) if (Voc_tandem > 0 and Jsc_tandem > 0) else np.nan
PCE_tandem = P_mpp / 100.0

results.append(["Tandem", f"{Jsc_tandem:.2f}", f"{Voc_tandem:.2f}", f"{FF_tandem*100:.2f}", f"{PCE_tandem*100:.2f}", f"{J_mpp:.2f}", f"{V_mpp:.2f}"])

# -----------------------------
# Ergebnisse anzeigen
# -----------------------------
df = pd.DataFrame(results, columns=["Zelle", "Jsc [mA/cm²]", "Voc [V]", "FF [%]", "PCE [%]", "Jmpp [mA/cm²]", "Vmpp [V]"])
st.table(df)

# -----------------------------
# Interaktive Plots (Plotly)
# -----------------------------
fig1 = go.Figure()
for i, V in enumerate(V_all):
    fig1.add_trace(go.Scatter(x=V, y=J_common, mode="lines", name=f"Zelle {i+1}"))
fig1.add_trace(go.Scatter(x=V_tandem, y=J_common, mode="lines", name="Tandem", line=dict(width=3)))
fig1.add_trace(go.Scatter(x=[V_mpp], y=[J_mpp], mode="markers", name="Tandem MPP",
                          marker=dict(color="red", size=10, symbol="x")))
fig1.update_layout(
    title="IV-Kennlinien",
    xaxis_title="Spannung [V]",
    yaxis_title="Stromdichte [mA/cm²]",
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)
