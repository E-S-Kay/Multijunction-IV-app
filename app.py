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
    q = 1.602176634e-19
    k = 1.380649e-23
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
    Jph = float(Jph_mA) / 1000.0
    J0  = float(J0_mA) / 1000.0
    cell = {"Jph": Jph, "J0": J0, "n": float(n), "Rs": float(Rs), "Rsh": float(Rsh), "T": float(T)}

    Voc = estimate_Voc(cell)
    V_vals = np.zeros_like(J_common, dtype=float)
    V_prev = Voc

    for i, JmA in enumerate(J_common):
        J = float(JmA) / 1000.0
        V_sol = None
        try:
            sol = root_scalar(lambda V: diode_equation_V(V, J, cell),
                              bracket=[-1.0, Voc + 1.5], method="bisect")
            if sol.converged:
                V_sol = sol.root
        except Exception:
            pass
        if V_sol is None:
            try:
                sol = fsolve(lambda V: diode_equation_V(V, J, cell), V_prev, maxfev=1000)
                V_sol = float(sol[0])
            except Exception:
                V_sol = float(V_prev)
        V_vals[i] = V_sol
        V_prev = V_sol

    P_plot = V_vals * J_common
    idx_mpp = int(np.nanargmax(P_plot))

    try:
        upper = max(1e-6, Jph_mA * 1.5)
        sol_j = root_scalar(lambda J: diode_equation_V(0.0, J/1000.0, cell),
                            bracket=[0.0, upper], method="bisect")
        Jsc_val = float(sol_j.root) if sol_j.converged else np.nan
    except Exception:
        Jsc_val = np.nan

    Vmpp = float(V_vals[idx_mpp])
    Jmpp = float(J_common[idx_mpp])
    Pmpp = float(P_plot[idx_mpp])

    return V_vals, P_plot, float(Voc), Vmpp, Jmpp, Pmpp, Jsc_val

def interpolate_Jsc_two_points_linreg(V, J):
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    if V.size < 2:
        return np.nan
    mask = ((V[:-1] <= 0.0) & (V[1:] >= 0.0)) | ((V[:-1] >= 0.0) & (V[1:] <= 0.0))
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return np.nan
    idx = int(idxs[0])
    V_pair = V[idx:idx+2]
    J_pair = J[idx:idx+2]
    if np.isclose(V_pair[0], V_pair[1]):
        return float(J_pair[0])
    slope, intercept, _, _, _ = stats.linregress(V_pair, J_pair)
    return float(intercept)

def calc_FF(Jsc, Voc, Jmpp, Vmpp):
    try:
        if np.isnan(Jsc) or Jsc == 0 or Voc == 0:
            return np.nan
        return (Jmpp * Vmpp) / (Jsc * Voc)
    except Exception:
        return np.nan

def to_float(text, default=0.0):
    try:
        return float(text.strip().replace(",", "."))
    except Exception:
        return float(default)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IV-Kennlinie: flexibel für 1–4 Teilzellen (Eindiodenmodell)")

num_cells = st.sidebar.selectbox("Anzahl der Teilzellen", [1, 2, 3, 4], index=1)

cells = []
for i in range(num_cells):
    st.sidebar.subheader(f"Zelle {i+1}")
    Jph = to_float(st.sidebar.text_input(f"Zelle {i+1}: Jph [mA/cm²]", value="30.0" if i == 0 else "20.0"))
    J0  = to_float(st.sidebar.text_input(f"Zelle {i+1}: J0 [mA/cm²]", value="1e-10" if i == 0 else "1e-12"))
    n   = to_float(st.sidebar.text_input(f"Zelle {i+1}: Idealfaktor n", value="1.0"))
    Rs  = to_float(st.sidebar.text_input(f"Zelle {i+1}: Rs [Ohm·cm²]", value="0.2"))
    Rsh = to_float(st.sidebar.text_input(f"Zelle {i+1}: Rsh [Ohm·cm²]", value="1000.0"))
    T   = to_float(st.sidebar.text_input(f"Zelle {i+1}: Temperatur T [K]", value="298.0"))
    cells.append({"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T})

J_common = np.linspace(0.0, max([c["Jph"] for c in cells]), 800)

V_all, P_all, rows = [], [], []
for i, c in enumerate(cells):
    V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv(c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"], J_common)
    V_all.append(V); P_all.append(P)
    FF = calc_FF(Jsc, Voc, Jmpp, Vmpp)
    rows.append({
        "Zelle": f"Zelle {i+1}",
        "Jsc": Jsc, "Voc": Voc, "FF": FF,
        "PCE": Pmpp, "Jmpp": Jmpp, "Vmpp": Vmpp
    })

V_stack = np.sum(np.vstack(V_all), axis=0)
P_stack = V_stack * J_common
idx_mpp_stack = int(np.nanargmax(P_stack))
Voc_stack = float(V_stack[0])
V_mpp_stack = float(V_stack[idx_mpp_stack])
J_mpp_stack = float(J_common[idx_mpp_stack])
P_mpp_stack = float(P_stack[idx_mpp_stack])
Jsc_stack = interpolate_Jsc_two_points_linreg(V_stack, J_common)
FF_stack = calc_FF(Jsc_stack, Voc_stack, J_mpp_stack, V_mpp_stack)

rows.append({
    "Zelle": "Stack", "Jsc": Jsc_stack, "Voc": Voc_stack,
    "FF": FF_stack, "PCE": P_mpp_stack,
    "Jmpp": J_mpp_stack, "Vmpp": V_mpp_stack
})

def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{x:.2f}"

df = pd.DataFrame({
    "Zelle": [r["Zelle"] for r in rows],
    "Jsc [mA/cm²]": [fmt(r["Jsc"]) for r in rows],
    "Voc [V]": [fmt(r["Voc"]) for r in rows],
    "FF [%]": [fmt(r["FF"] * 100.0) if (r["FF"] is not None and not np.isnan(r["FF"])) else "NaN" for r in rows],
    "PCE [%]": [fmt(r["PCE"]) for r in rows],
    "Jmpp [mA/cm²]": [fmt(r["Jmpp"]) for r in rows],
    "Vmpp [V]": [fmt(r["Vmpp"]) for r in rows],
})

st.write("### Ergebnisse")
st.table(df)

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()
for i, V in enumerate(V_all):
    fig.add_trace(go.Scatter(x=V, y=J_common, mode="lines", name=f"Zelle {i+1}"))
fig.add_trace(go.Scatter(x=V_stack, y=J_common, mode="lines", name="Stack", line=dict(width=3)))
fig.add_trace(go.Scatter(x=[V_mpp_stack], y=[J_mpp_stack], mode="markers", name="Stack MPP",
                         marker=dict(color="red", size=10, symbol="x")))

# Linien bei x=0 und y=0
fig.add_shape(type="line", x0=0, x1=0, y0=min(J_common), y1=max(J_common),
              line=dict(color="gray", dash="dash"))
fig.add_shape(type="line", x0=min(V_stack), x1=max(V_stack), y0=0, y1=0,
              line=dict(color="gray", dash="dash"))

fig.update_layout(
    title="IV-Kennlinien",
    xaxis_title="Spannung [V]",
    yaxis_title="Stromdichte [mA/cm²]",
    hovermode="x unified"
)

# X-Achse von -0.2 bis VOC_stack + 0.1
fig.update_xaxes(range=[-0.2, Voc_stack + 0.1])

st.plotly_chart(fig, use_container_width=True)
