# app.py - Tandem IV (beide Teilzellen auf gleichem J auswerten)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import root_scalar, fsolve

st.set_page_config(page_title="Tandem IV - gleiche J-Werte", layout="centered")

# -------------------------
# Numerische Hilfsfunktionen
# -------------------------
q = 1.602176634e-19  # C
k = 1.380649e-23     # J/K

def safe_exp(x):
    return np.exp(np.clip(x, -700, 700))

def diode_equation_V(V, J, cell):
    """
    Diode-Glgleichung in der Form f(V) = 0 für gegebenen J (alle internen Werte in SI):
      f(V) = J - [Jph - J0*(exp(q*(V + J*Rs)/(n*k*T)) - 1) - (V + J*Rs)/Rsh]
    J in A/cm^2, Rs in Ohm*cm^2, Rsh in Ohm*cm^2, V in V
    """
    arg = q * (V + J * cell["Rs"]) / (cell["n"] * k * cell["T"])
    exp_term = safe_exp(arg)
    return J - (cell["Jph"] - cell["J0"] * (exp_term - 1.0) - (V + J * cell["Rs"]) / cell["Rsh"])

def estimate_Voc(cell):
    """
    Versucht Voc numerisch zu finden. Falls das fehlschlägt, benutzt
    vereinfachte analytische Näherung (ideal, ohne Rs/Rsh).
    """
    try:
        sol = root_scalar(lambda V: diode_equation_V(V, 0.0, cell),
                          bracket=[-0.5, 2.0], method="bisect", xtol=1e-8)
        if sol.converged:
            return float(sol.root)
    except Exception:
        pass
    # fallback: ideale Näherung (Achtung: nur sinnvoll wenn J0>0)
    J0safe = max(cell["J0"], 1e-30)
    try:
        voc_approx = (cell["n"] * k * cell["T"] / q) * np.log(max(cell["Jph"]/J0safe, 1.0) + 1.0)
        return float(voc_approx)
    except Exception:
        return 0.6

def solve_voltage_for_J(J, cell, Voc_estimate, V_prev):
    """
    Löse V für gegebenen J:
    - zuerst versuchen wir root_scalar mit sicherem Intervall [Vmin, Vmax] (bisection)
    - falls das nicht möglich ist (kein Vorzeichenwechsel), Fallback auf fsolve mit Startwert V_prev
    """
    V_min = -2.0
    V_max = max(Voc_estimate + 1.0, 1.0)
    try:
        fmin = diode_equation_V(V_min, J, cell)
        fmax = diode_equation_V(V_max, J, cell)
        if fmin * fmax < 0:
            sol = root_scalar(lambda V: diode_equation_V(V, J, cell),
                              bracket=[V_min, V_max], method="bisect", xtol=1e-8)
            if sol.converged:
                return float(sol.root)
    except Exception:
        pass

    # fallback: fsolve mit näherungswert
    guess = V_prev
    # fallback guess if V_prev is None or nan
    if guess is None or not np.isfinite(guess):
        # lineare Schätzung: Voc*(1 - J/Jph) falls Jph>0
        if cell["Jph"] > 0:
            guess = Voc_estimate * max(0.0, 1.0 - J / cell["Jph"])
        else:
            guess = 0.0
    try:
        sol = fsolve(lambda V: diode_equation_V(V, J, cell), guess, xtol=1e-10, maxfev=200)
        return float(sol[0])
    except Exception:
        return float(guess)

def compute_VJ_for_cell(cell, J_common):
    """
    Für ein gegebenes cell-Dict (Jph,J0,n,Rs,Rsh,T in SI-Einheiten) berechne V(J)
    für alle J in J_common (array, A/cm^2). Liefert V_vals und Voc.
    """
    Voc = estimate_Voc(cell)
    V_vals = np.zeros_like(J_common)
    V_prev = Voc
    for i, J in enumerate(J_common):
        V_sol = solve_voltage_for_J(J, cell, Voc, V_prev)
        V_vals[i] = V_sol
        V_prev = V_sol
    return V_vals, Voc

# -------------------------
# Input-parsing (alle als Text)
# -------------------------
st.title("Tandem-IV: 2 Teilzellen — Spannungen bei gleichen Strömen addieren")
st.markdown("Alle Eingaben als Text (Wissenschaftliche Notation z.B. `1e-10` möglich). "
            "Einheiten: J in mA/cm², Rs/Rsh in Ω·cm², T in K.")

def parse_float_text(label, default):
    s = st.sidebar.text_input(label, value=str(default))
    try:
        val = float(s)
        return val
    except Exception:
        st.sidebar.error(f"Ungültige Eingabe für '{label}'. Benutze z.B. 1e-10 oder 0.0001")
        return float(default)

def parse_int_text(label, default):
    s = st.sidebar.text_input(label, value=str(default))
    try:
        val = int(float(s))
        if val < 2:
            return default
        return val
    except Exception:
        st.sidebar.error(f"Ungültige Eingabe für '{label}'. Ganzzahl erwartet.")
        return default

st.sidebar.header("Zelle 1 (oben)")
Jph1_mA = parse_float_text("Z1: Photostrom Jph [mA/cm²]", 30.0)
J01_mA  = parse_float_text("Z1: Sättigungsstrom J0 [mA/cm²]", 1e-10)
n1      = parse_float_text("Z1: Idealfaktor n", 1.0)
Rs1     = parse_float_text("Z1: Serienwiderstand Rs [Ohm·cm²]", 0.2)
Rsh1    = parse_float_text("Z1: Parallelwiderstand Rsh [Ohm·cm²]", 1000.0)
T1      = parse_float_text("Z1: Temperatur T [K]", 298.0)

st.sidebar.header("Zelle 2 (unten)")
Jph2_mA = parse_float_text("Z2: Photostrom Jph [mA/cm²]", 20.0)
J02_mA  = parse_float_text("Z2: Sättigungsstrom J0 [mA/cm²]", 1e-12)
n2      = parse_float_text("Z2: Idealfaktor n", 1.0)
Rs2     = parse_float_text("Z2: Serienwiderstand Rs [Ohm·cm²]", 0.2)
Rsh2    = parse_float_text("Z2: Parallelwiderstand Rsh [Ohm·cm²]", 1000.0)
T2      = parse_float_text("Z2: Temperatur T [K]", 298.0)

Npts = parse_int_text("Anzahl Punkte für J-Gitter", 400)

# -------------------------
# Vorverarbeitung / Umrechnung
# -------------------------
# Umrechnung mA/cm^2 -> A/cm^2 intern
Jph1 = max(0.0, Jph1_mA / 1000.0)
J01  = max(0.0, J01_mA  / 1000.0)
Jph2 = max(0.0, Jph2_mA / 1000.0)
J02  = max(0.0, J02_mA  / 1000.0)

cell1 = {"Jph": Jph1, "J0": J01, "n": float(n1), "Rs": float(Rs1), "Rsh": float(Rsh1), "T": float(T1)}
cell2 = {"Jph": Jph2, "J0": J02, "n": float(n2), "Rs": float(Rs2), "Rsh": float(Rsh2), "T": float(T2)}

# Gemeinsames J-Gitter: von 0 bis min(Jph1,Jph2)
Jmax_common = min(cell1["Jph"], cell2["Jph"])
if Jmax_common <= 0:
    st.warning("Mindestens eine Zelle hat Jph = 0 (oder negativ). Bitte realistische Photoströme eingeben.")
J_common = np.linspace(0.0, max(0.0, Jmax_common), max(2, int(Npts)))  # A/cm^2

# -------------------------
# Berechnung (beide Zellen auf Am gemeinsamen J)
# -------------------------
with st.spinner("Berechne V(J) für beide Teilzellen..."):
    V1_vals, Voc1 = compute_VJ_for_cell(cell1, J_common)
    V2_vals, Voc2 = compute_VJ_for_cell(cell2, J_common)

# Tandem: Spannung addieren bei gleichen J
V_tandem = V1_vals + V2_vals
J_plot = J_common * 1000.0          # zurück zu mA/cm^2 für Anzeige
P_tandem = V_tandem * J_plot        # mW/cm^2

# MPP bestimmen
if np.all(np.isnan(P_tandem)):
    st.error("Fehler: P_tandem enthält nur NaN. Überprüfe Parameter.")
else:
    idx_mpp = int(np.nanargmax(P_tandem))
    V_mpp = float(V_tandem[idx_mpp])
    J_mpp = float(J_plot[idx_mpp])
    P_mpp = float(P_tandem[idx_mpp])
    Voc_tandem = float(V_tandem[0])  # bei J=0

# -------------------------
# Anzeige Ergebnisse & Plots
# -------------------------
st.subheader("Ergebnisse (Kurz)")
col1, col2 = st.columns(2)
col1.metric("Zelle1 Voc [V]", f"{Voc1:.4f}")
col2.metric("Zelle2 Voc [V]", f"{Voc2:.4f}")
st.metric("Tandem Voc [V]", f"{Voc_tandem:.4f}")

st.write(f"**Tandem MPP**: V = {V_mpp:.4f} V, J = {J_mpp:.4f} mA/cm², P = {P_mpp:.4f} mW/cm²")

# IV-Plot (Teilzellen + Tandem)
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(V1_vals, J_plot, label="Zelle 1")
ax1.plot(V2_vals, J_plot, label="Zelle 2")
ax1.plot(V_tandem, J_plot, label="Tandem (V1+V2)", linewidth=2)
ax1.scatter([V_mpp], [J_mpp], color="red", label="Tandem MPP")
ax1.set_xlabel("Spannung [V]")
ax1.set_ylabel("Stromdichte [mA/cm²]")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# P-V Kurve Tandem
fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.plot(V_tandem, P_tandem, label="Tandem P-V")
ax2.scatter([V_mpp], [P_mpp], color="red", label="MPP")
ax2.set_xlabel("Spannung [V]")
ax2.set_ylabel("Leistung [mW/cm²]")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# -------------------------
# Daten-Tabelle & Download
# -------------------------
df = pd.DataFrame({
    "J [mA/cm^2]": J_plot,
    "V1 [V]": V1_vals,
    "V2 [V]": V2_vals,
    "V_tandem [V]": V_tandem,
    "P_tandem [mW/cm^2]": P_tandem
})
st.subheader("Rohdaten")
st.dataframe(df.style.format({
    "J [mA/cm^2]": "{:.6f}",
    "V1 [V]": "{:.6f}",
    "V2 [V]": "{:.6f}",
    "V_tandem [V]": "{:.6f}",
    "P_tandem [mW/cm^2]": "{:.6f}"
}))

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("CSV herunterladen (Tandem)", data=csv, file_name="tandem_iv.csv", mime="text/csv")

st.markdown("---")
st.caption("Hinweis: Berechnungen intern in A/cm². Die Anzeige verwendet mA/cm² (J) und mW/cm² (P).")
