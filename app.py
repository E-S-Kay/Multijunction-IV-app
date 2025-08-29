import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar

# ---------------------
# Hilfsfunktionen
# ---------------------
def safe_exp(x):
    """Numerisch stabile Exponentialfunktion (vermeidet overflow)."""
    x = np.clip(x, -700, 700)
    return np.exp(x)

def diode_current_from_V(V, cell, J):
    """
    Rückgabe: f(V) = J - [Jph - J0*(exp(...) - 1) - (V + J*Rs)/Rsh]
    Nullstelle f(V)=0 => V ist Lösung für gegebenen J.
    """
    q = 1.602e-19
    k = 1.381e-23
    T = cell["T"]
    arg = q * (V + J * cell["Rs"]) / (cell["n"] * k * T)
    exp_term = safe_exp(arg)
    return J - (cell["Jph"] - cell["J0"] * (exp_term - 1) - (V + J * cell["Rs"]) / cell["Rsh"])

def solve_voltage_for_J(J, cell, V_guess=None, Voc_estimate=None):
    """
    Versucht zuerst root_scalar mit einem sinnvollen Intervall [Vmin, Vmax].
    Falls das fehlschlägt, fällt es auf fsolve zurück.
    """
    # Set interval for bracket search
    Vmin = -5.0  # ausreichend tief (für numerische Robustheit)
    Vmax = max(1.0, (Voc_estimate or 0.6) + 2.0)

    # Ensure function signs differ at interval boundaries for root_scalar
    fmin = diode_current_from_V(Vmin, cell, J)
    fmax = diode_current_from_V(Vmax, cell, J)

    if fmin * fmax < 0:
        try:
            sol = root_scalar(lambda V: diode_current_from_V(V, cell, J),
                              bracket=[Vmin, Vmax], method='bisect', xtol=1e-6)
            if sol.converged:
                return sol.root
        except Exception:
            pass

    # fallback: fsolve with a guess
    if V_guess is None:
        # reasonable initial guess: linear interpolation using Jph (short circuit) and Voc_estimate
        if Voc_estimate is not None:
            V_guess = Voc_estimate * (1 - J / max(1e-12, cell["Jph"]))
        else:
            V_guess = 0.5
    try:
        sol = fsolve(lambda V: diode_current_from_V(V, cell, J), V_guess, xtol=1e-8, maxfev=200)
        return float(sol[0])
    except Exception:
        return float(V_guess)

def estimate_Voc(cell):
    """Schätzt die Leerlaufspannung Voc durch Lösen der Gleichung bei J=0."""
    try:
        # Use root_scalar with bracket
        Vmin = -1.0
        Vmax = 2.0
        fmin = diode_current_from_V(Vmin, cell, 0.0)
        fmax = diode_current_from_V(Vmax, cell, 0.0)
        if fmin * fmax < 0:
            sol = root_scalar(lambda V: diode_current_from_V(V, cell, 0.0), bracket=[Vmin, Vmax], method='bisect', xtol=1e-6)
            if sol.converged:
                return sol.root
        # fallback: enlarge bracket
        Vmin, Vmax = -5.0, 5.0
        sol = root_scalar(lambda V: diode_current_from_V(V, cell, 0.0), bracket=[Vmin, Vmax], method='bisect', xtol=1e-6)
        if sol.converged:
            return sol.root
    except Exception:
        pass
    # if all fails, approximate
    return 0.6

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Single-Junction IV-Kennlinie", layout="centered")
st.title("🔋 Single-Junction Solarzelle — IV & P-V Berechnung")

st.markdown(
    "Dieses Tool berechnet die IV-Kennlinie einer einzelnen Solarzellen-Junction mithilfe des Eindiodenmodells."
    " Eingaben sind in mA/cm² für Ströme und Ohm·cm² für Widerstände."
)

st.sidebar.header("Zellparameter (Einfach)")
Jph = st.sidebar.number_input("Photostrom J_ph [mA/cm²]", value=30.0, format="%.6g")
J0 = st.sidebar.number_input("Sättigungsstrom J_0 [mA/cm²]", value=1e-10, format="%.1e")
n  = st.sidebar.number_input("Idealfaktor n", value=1.0, format="%.3f")
Rs = st.sidebar.number_input("Serienwiderstand R_s [Ohm·cm²]", value=0.2, format="%.6g")
Rsh= st.sidebar.number_input("Parallelwiderstand R_sh [Ohm·cm²]", value=1000.0, format="%.6g")
T  = st.sidebar.number_input("Temperatur T [K]", value=298.0, format="%.3f")
N  = st.sidebar.number_input("Punkte für Kurve", min_value=50, max_value=5000, value=500, step=50)

cell = {"Jph": float(Jph), "J0": float(J0), "n": float(n), "Rs": float(Rs), "Rsh": float(Rsh), "T": float(T)}

st.sidebar.markdown("---")
st.sidebar.markdown("Hinweis: Einheiten: J in mA/cm², V in V, P in mW/cm²")

if st.button("Berechne IV & P-V Kurve"):
    # Stromwerte (mA/cm²) von 0 (Kurzschluss) bis Jph (Kurzschlussstrom)
    J_values = np.linspace(0.0, max(0.0, cell["Jph"]), int(N))

    # Erst Voc schätzen (für bessere Anfangswerte)
    Voc = estimate_Voc(cell)

    # Berechne V(J)
    V_values = np.zeros_like(J_values)
    V_prev = Voc
    for i, J in enumerate(J_values):
        # guess based on previous value for convergence
        V_guess = V_prev if i>0 else Voc * (1 - J / max(1e-12, cell["Jph"]))
        V_sol = solve_voltage_for_J(J, cell, V_guess=V_guess, Voc_estimate=Voc)
        V_values[i] = V_sol
        V_prev = V_sol

    # Leistung (mW/cm²): P = V [V] * J [mA/cm²] -> mW/cm²
    P_values = V_values * J_values

    # Finde MPP
    idx_mpp = int(np.nanargmax(P_values))
    J_mpp = J_values[idx_mpp]
    V_mpp = V_values[idx_mpp]
    P_mpp = P_values[idx_mpp]

    # Plot IV
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(V_values, J_values, label="IV", linewidth=2)
    ax1.scatter([V_mpp], [J_mpp], color="red", label=f"MPP @ V={V_mpp:.3f} V, J={J_mpp:.3f} mA/cm²")
    ax1.set_xlabel("Spannung V [V]")
    ax1.set_ylabel("Stromdichte J [mA/cm²]")
    ax1.grid(True)
    ax1.legend()

    # Plot P-V
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(V_values, P_values, label="P-V", linewidth=2)
    ax2.scatter([V_mpp], [P_mpp], color="red", label=f"MPP: P={P_mpp:.3f} mW/cm²")
    ax2.set_xlabel("Spannung V [V]")
    ax2.set_ylabel("Leistung P [mW/cm²]")
    ax2.grid(True)
    ax2.legend()

    st.subheader("Ergebnisse")
    st.write(f"Leerlaufspannung Voc (geschätzt): {Voc:.4f} V")
    st.write(f"Maximum Power Point (MPP): V = {V_mpp:.4f} V, J = {J_mpp:.4f} mA/cm², P = {P_mpp:.4f} mW/cm²")

    st.pyplot(fig1)
    st.pyplot(fig2)

    # Tabelle und CSV-Download
    df = pd.DataFrame({
        "V [V]": V_values,
        "J [mA/cm^2]": J_values,
        "P [mW/cm^2]": P_values
    })
    st.subheader("Daten")
    st.dataframe(df.style.format({"V [V]":"{:.6f}", "J [mA/cm^2]":"{:.6f}", "P [mW/cm^2]":"{:.6f}"}))

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="CSV herunterladen", data=csv, file_name="single_junction_iv.csv", mime="text/csv")

else:
    st.info("Drücke 'Berechne IV & P-V Kurve', um die Kurven zu berechnen.")

st.markdown("---")
st.markdown("© 2025 – Single-Junction IV Tool")
