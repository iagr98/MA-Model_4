# Abscheider-Modell
Dieser Code beinhaltet die Implementierung eines dynamischen 1D-Abscheidermodells. Der Code ist objektorientiert aufgebaut; zum Ausführen ist die Datei **`sim_run.py`** zu verwenden. Messwerte von Henschke sowie Stoffdaten werden aus den Excel-Dateien im *Input*-Ordner eingelesen. Nachfolgend ein Überblick über die wichtigsten Dateien:

- **`sim_run.py`** – Hauptskript zur Ausführung der Simulation
- **`sim_model.py`** – enthält die Berechnungsvorschriften und Plot-Funktionen
- **`sim_parameters.py`** – enthält Parameter für die Berechnungen und liest Messwerte sowie Stoffdaten aus den Excel-Dateien im *Input*-Ordner ein
- **`helper_functions.py`** – beinhaltet Hilfsfunktionen für die Berechnungen

### Anmerkungen
- **Terminierungskriterium:** Die Simulation läuft, bis der Solver `solve_ivp` entweder den dicht gepackten Sichtbereich (DGTS) vollständig gefüllt hat oder ein stationärer Zustand erreicht wurde (Volumen in einem Element des DGTS wird null).
- Der Parameter **`exponent`** wird in der Datei **`sim_run.py`** abhängig vom Experimenttyp der Simulation zugewiesen. Die Exponenten wurden im Rahmen einer Validierungsanalyse bestimmt.
- Alle konvektiven Geschwindigkeiten $u_\mathrm{dis}, u_c$ und $u_d$ werden in der Methode **`velocities`** in der Datei **`sim_model.py`** berechnet.
