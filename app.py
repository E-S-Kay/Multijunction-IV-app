import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

st.set_page_config(page_title="Tandemsolarzellen IV-Kennlinie", layout="centered")
st.title("🔋 IV-Kennlinie einer Tandemsolarzelle")
st.markdown("Berechnung der IV-Kennlinie einer Tandemsolarzelle basierend auf dem Eindiodenmodell.")

st.sidebar.header("📥 Eingabeparameter")

def get_cell_params(name):
    st.sidebar.subheader(f"{name} Teilzelle")
    Jph = st.sidebar.number_input(f"{name} Photostrom $J_{{ph}}$ [mA/cm²]", value=30.0)
    J0 = st.sidebar.number_input(f"{name} Sättigungsstrom $J_0$ [mA/cm²]", value=1e-10, format="%.1e")
    n = st.sidebar.number_input(f"{name} Idealiätsfaktor $n$", value=1.0)
    Rs = st.sidebar.number_input(f"{name} Serienwiderstand $R_s$ [Ohm·cm²]", value=0.2)
    Rsh = st.sidebar.number_input(f"{name} Parallelwiderstand $R_{{sh}}$ [Ohm·cm²]", value=1000.0)
    return dict(Jph=Jph, J0=J0, n=n, Rs=Rs, Rsh=Rsh)

cell1 = get_cell_params("Obere")
cell2 = get_cell_params("Untere")

q = 1.602e-19
k = 1.381e-23
T = 298

st.subheader("📉 IV-Kennlinie berechnen")

J_values = np.linspace(0, min(cell1["Jph"], cell2["Jph"]), 300)

def diode_voltage(J, cell):
    def equation(V):
        term1 = cell["Jph"]
        term2 = cell["J0"] * (np.exp(q * (V + J * cell["Rs"]) / (cell["n"] * k * T)) - 1)
        term3 = (V + J * cell["Rs"]) / cell["Rsh"]
        return J - (term1 - term2 - term3)
    V_guess = 0.5
    V_solution = fsolve(equation, V_guess)
    return V_solution[0]

V1 = np.array([diode_voltage(J, cell1) for J in J_values])
V2 = np.array([diode_voltage(J, cell2) for J in J_values])
V_tandem = V1 + V2

st.subheader("📊 Diagramm")

fig, ax = plt.subplots()
ax.plot(V_tandem, J_values, label="Tandemzelle", color="black")
ax.plot(V1, J_values, '--', label="Obere Zelle", alpha=0.7)
ax.plot(V2, J_values, '--', label="Untere Zelle", alpha=0.7)
ax.set_xlabel("Spannung [V]")
ax.set_ylabel("Stromdichte [mA/cm²]")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("© 2025 – Solar IV Tool · Erstellt mit Python & Streamlit")
