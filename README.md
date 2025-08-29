# 🔋 Tandemsolarzellen IV-Kennlinie (Streamlit App)

Diese App berechnet und visualisiert die IV-Kennlinie einer Tandemsolarzelle mittels Eindiodengleichung für beide Teilzellen.

## ✅ Funktionen
- Eingabeparameter pro Zelle: Photostrom, Sättigungsstrom, Idealfaktor, Serien- & Parallelwiderstand
- Darstellung der IV-Kennlinie beider Teilzellen + der Tandemzelle
- Interaktive Oberfläche, sofortige grafische Ausgabe

## 🚀 Online verwenden (Streamlit Cloud)
1. Erstelle ein GitHub-Konto
2. Erstelle ein neues Repository (z. B. `tandem-iv-app`)
3. Füge die Dateien `app.py` und `requirements.txt` ein
4. Gehe auf https://streamlit.io/cloud
5. Logge dich mit GitHub ein und klicke **"New app"**
6. Wähle dein Repo, den Branch (z. B. `main`) und Datei `app.py`
7. Klicke auf **Deploy**

👉 Deine App ist nun unter einer öffentlichen URL erreichbar wie:  
`https://dein-name-tandem-iv-app.streamlit.app`

## 📥 Beispielparameter
- Jph = 30 mA/cm²
- J0 = 1e-10 mA/cm²
- n = 1.0
- Rs = 0.2 Ohm·cm²
- Rsh = 1000 Ohm·cm²
