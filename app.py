import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -----------------------------
# Hilfsfunktionen
# -----------------------------
def fmt(x, ndigits=2):
    try:
        return f"{x:.{ndigits}f}"
    except Exception:
        return "NaN"

def interpolate_Jsc_two_points_linreg(V, J):
    idx = np.where(np.diff(np.sign(J)))[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    x1, y1 = V[i], J[i]
    x2, y2 = V[i+1], J[i+1]
    return np.interp(0, [x1, x2], [y1, y2])

def calc_FF(Jsc, Voc, Jmpp, Vmpp):
    if Voc == 0 or Jsc == 0:
        return float("nan")
    return (Jmpp * Vmpp) / (Voc * Jsc)

# -----------------------------
# Farben
# -----------------------------
pastel_colors = ["#AFCBFF", "#FFCBAF", "#CBAFFF", "#AFFFCB"]  # beliebig erweiterbar
stack_color = "black"

# -----------------------------
# Eingabeparameter
# -----------------------------
st.sidebar.header("Eingabeparameter")

num_cells = st.sidebar.number_input("Anzahl der Zellen", min_value=1, max_value=4, value=2, step=1)

cells = []
for i in range(num_cells):
    color = pastel_colors[i % len(pastel_colors)]
    # CSS für dieses Eingabefeld einfügen
    st.sidebar.markdown(
        f"""
        <style>
        div[data-testid="stTextInput"] input[data-baseweb="input"]:nth-of-type({i+1}) {{
            background-color: {color};
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    Jph = float(st.sidebar.text_input(f"Zelle {i+1}: Jph [mA/cm²]", value="30.0", key=f"Jph{i}"))
    Jo = float(st.sidebar.text_input(f"Zelle {i+1}: Jo [mA/cm²]", value="1e-9", key=f"Jo{i}"))
    n = float(st.sidebar.text_input(f"Zelle {i+1}: Ideality n", value="1.0", key=f"n{i}"))
    cells.append({"Jph": Jph, "Jo": Jo, "n": n, "color": color})

# -----------------------------
# Berechnungen
# -----------------------------
V_common = np.linspace(0, 2, 500)
rows = []
V_all = []

for i, cell in enumerate(cells):
    Jph, Jo, n = cell["Jph"], cell["Jo"], cell["n"]
    Vt = 0.02585  # thermische Spannung
    J = Jph - Jo * (np.exp(V_common / (n * Vt)) - 1)

    V = V_common
    Jsc = Jph
    Voc = n * Vt * np.log(Jph / Jo + 1)

    P = V * J
    idx_mpp = np.argmax(P)
    V_mpp, J_mpp, P_mpp = V[idx_mpp], J[idx_mpp], P[idx_mpp]
    FF = calc_FF(Jsc, Voc, J_mpp, V_mpp)

    rows.append({
        "Jsc": Jsc, "Voc": Voc,
        "FF": FF, "PCE": P_mpp,
        "Jmpp": J_mpp, "Vmpp": V_mpp,
        "color": cell["color"]
    })
    V_all.append(V)

# Stack berechnen, falls mehrere Zellen
if num_cells > 1:
    V_stack = np.sum(np.vstack(V_all), axis=0)
    P_stack = V_stack * J
    idx_mpp_stack = np.argmax(P_stack)
    Voc_stack = float(np.sum([r["Voc"] for r in rows]))
    V_mpp_stack = float(V_stack[idx_mpp_stack])
    J_mpp_stack = float(J[idx_mpp_stack])
    P_mpp_stack = float(P_stack[idx_mpp_stack])
    Jsc_stack = min([r["Jsc"] for r in rows])
    FF_stack = calc_FF(Jsc_stack, Voc_stack, J_mpp_stack, V_mpp_stack)

    rows.append({
        "Jsc": Jsc_stack, "Voc": Voc_stack,
        "FF": FF_stack, "PCE": P_mpp_stack,
        "Jmpp": J_mpp_stack, "Vmpp": V_mpp_stack,
        "color": "transparent"  # Stack transparent
    })

# -----------------------------
# Ergebnistabelle
# -----------------------------
df = pd.DataFrame({
    "Jsc [mA/cm²]": [fmt(r["Jsc"], 2) for r in rows],
    "Voc [V]": [fmt(r["Voc"], 3) for r in rows],
    "FF [%]": [fmt(r["FF"]*100.0, 2) if (r["FF"] is not None and not np.isnan(r["FF"])) else "NaN" for r in rows],
    "PCE [%]": [fmt(r["PCE"], 2) for r in rows],
    "Jmpp [mA/cm²]": [fmt(r["Jmpp"], 2) for r in rows],
    "Vmpp [V]": [fmt(r["Vmpp"], 3) for r in rows],
    "color": [r["color"] for r in rows]
})

df_display = df.drop(columns=["color"])
row_colors = df["color"].tolist()

def highlight_rows(row):
    color = row_colors[row.name]
    return [f'background-color: {color}']*len(row)

st.write("### Ergebnisse")
st.dataframe(df_display.style.apply(highlight_rows, axis=1))

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()

for i, (V, cell) in enumerate(zip(V_all, cells)):
    Jph, Jo, n = cell["Jph"], cell["Jo"], cell["n"]
    Vt = 0.02585
    J = Jph - Jo * (np.exp(V / (n * Vt)) - 1)
    fig.add_trace(go.Scatter(x=V, y=J, mode="lines", name=f"Zelle {i+1}",
                             line=dict(color=cell["color"], width=2)))

if num_cells > 1:
    fig.add_trace(go.Scatter(x=V_stack, y=J, mode="lines", name="Stack",
                             line=dict(color=stack_color, width=4)))

fig.add_vline(x=0, line=dict(color="gray", dash="dash"))
fig.add_hline(y=0, line=dict(color="gray", dash="dash"))

if num_cells > 1:
    x_max = Voc_stack + 0.1
else:
    x_max = rows[0]["Voc"] + 0.1

fig.update_xaxes(range=[-0.2, x_max])
fig.update_layout(
    title="IV-Kennlinien",
    xaxis_title="Spannung [V]",
    yaxis_title="Stromdichte [mA/cm²]",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
