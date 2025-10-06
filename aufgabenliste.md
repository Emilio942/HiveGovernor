# Aufgabenliste (.mb): Swarm-MOI Monitor & Governor
_Status: OPEN • Version 1.0_

Ziel: Kleine## Phase 3 –  - **Done wenn:** Tests + Status ok. ✅

11. [DONE] **T31: Kurzreporter**
  - Datei: `swarm_moi/reporter.py`  
    `build_report(C, loop_pairs, labels)` → Dict mit:
    - Top-Gruppen (Größe, Vertreter), Top-Loop-Paare, Inhibitions-History-Stub
  - **Tests:** `tests/test_reporter_schema.py` (Schema/Schlüssel vorhanden, Werte sinnig) ✅
  - **Done wenn:** Tests + Status ok. ✅

12. [OPEN] **T32: Minimal-Console-Report**luster & Reports
10. [DONE] **T30: Clustering**
  - Datei: `swarm_moi/clustering.py`  
    Funktion `cluster_from_C(C) -> labels` (einfach: spektral oder „schwellwert-basiert"; keine schweren Libs).
  - **Tests:** `tests/test_clustering_simple.py` (klare Blöcke → korrekte Labels) ✅
  - **Done wenn:** Tests + Status ok. ✅

11. [OPEN] **T31: Kurzreporter**-Module koordinieren, **aktive Gruppen erkennen**, **Ping-Pong/Endlosschleifen verhindern** (Governor: Hysterese, Inhibition, Refraktärzeit) und **kurze, verlässliche Statusberichte** liefern – ohne viel Lesen.

---

## Globale Regeln (für Mensch & KI)
- **GR-1 (Kleine Diffs):** Max. 200 geänderte Zeilen, max. 3 Dateien pro Task/PR.
- **GR-2 (Guarded Regions):** Codeänderungen **nur** zwischen:
>>> BEGIN:AI_EDIT
>>> END:AI_EDIT
markdown
Copy code
- **GR-3 (Plan-Pflicht):** Vor jeder Änderung `plans/<ID>.yaml` schreiben (Intent, Dateien, Tests).
- **GR-4 (Tests zuerst):** Jede Codeänderung bringt **mind. 1 neuen Unit-Test**.
- **GR-5 (Status):** Fortschritt nur via `tools/status.sh` prüfen (keine freien Romane).
- **GR-6 (Ich vermute):** Wo Annahmen nötig sind, schreibe „Ich vermute: …“.

---

## Phase 0 – Projekt anlegen
1. [DONE] **T00: Grundstruktur & Werkzeug**
 - Dateien/Ordner:  
   `swarm_moi/ __init__.py`, `tests/`, `examples/`, `plans/`, `tools/`, `pyproject.toml` oder `requirements.txt`
 - Mindest-Deps: `torch`, `pytest`, `pytest-cov`, `ruff`, `mypy`, `diff-cover`, `ripgrep (rg)`, `mutmut`
 - **Tests hinzufügen:** `tests/test_repo_structure.py` (Existenz der Ordner/Dateien)
 - **Done wenn:** `pytest -q` grün; `ruff check` & `mypy` laufen ohne Fehler. ✅

2. [OPEN] **T01: Statusskript**
 - Datei: `tools/status.sh` (ausführbar) mit:
   - `pytest -q`, Coverage + Diff-Coverage (≥85%), `mypy --strict`, `ruff check --fix`, `mutmut run --use-coverage` (nur Ergebnisse zeigen)
 - **Tests:** `tests/test_status_smoke.py` (führt Skript subprocess-weise aus, erwartet Exit-Code 0)
 - **Done wenn:** `bash tools/status.sh` → Exit 0 & kurze Zusammenfassung.

3. [DONE] **T02: Guard-Enforcer (Pre-Check)**
 - Datei: `tools/enforce_guards.py`
   - Bricht, wenn Änderungen außerhalb der Marker erfolgen.
   - Bricht, wenn Diff > GR-1.
 - **Tests:** `tests/test_enforcer_simulated_diff.py` (simuliert ok/fail-Diffs) ✅
 - **Done wenn:** Enforcer auf Repo-Head läuft und beides erkennt (ok/fail). ✅

---

## Phase 1 – Router & Monitor (Minimalfunktion)
4. [OPEN] **T10: Router-Skelett**
 - Datei: `swarm_moi/router.py`  
   Klasse `HysteresisRouter(n_experts, k, tau, hysteresis)`
   - Platzhalter-Linearprojektor, Softmax(τ), Hysterese (über letzte Maske).
   - **Nur** innerhalb Guard-Region implementieren.
 - **Tests:**  
   `tests/test_router_init.py` (Shapes, k-Selektion),  
   `tests/test_router_hysteresis_effect.py` (leichte Rauschänderung → stabile Auswahl)
 - **Done wenn:** Tests grün, `tools/status.sh` grün.

5. [OPEN] **T11: Monitor (Ko-Aktivationen)**
 - Datei: `swarm_moi/monitor.py`  
   Klasse `Monitor(n_experts, ema)` mit `step(mask)` und `co_matrix()`
   - `C ← (1-α)C + α·(mᵀm)`; Normierung optional.
 - **Tests:**  
   `tests/test_monitor_coactivations.py` (synthetische Masken → erwartete C-Muster)
 - **Done wenn:** Tests + Status ok.

6. [OPEN] **T12: Loop-Heuristik**
 - In `monitor.py`: `loop_pairs(thresh)` → symmetrische starke Paare (i↔j) erkennen.
 - **Tests:** `tests/test_monitor_loop_pairs.py` (künstliche C → korrekte Top-Paare)
 - **Done wenn:** Tests + Status ok.

---

## Phase 2 – Governor (Anti-Loop)
7. [OPEN] **T20: Governor-Grundlogik**
 - Datei: `swarm_moi/governor.py`  
   Klasse `Governor(router, cooldown, gamma)`:
   - Refraktärzeit/Cooldown je Experte; Inhibition `router.inhibit` (multiplikativ).
   - Methode `apply(loop_pairs)` dämpft Top-Paar(e).
 - **Tests:**  
   `tests/test_governor_inhibition.py` (γ wirkt),  
   `tests/test_governor_cooldown.py` (Cooldown zählt sauber runter)
 - **Done wenn:** Tests + Status ok.

8. [OPEN] **T21: Router-Hook für Inhibition**
 - `router.py`: Inferenzpfad multipliziert `p` mit `router.inhibit` + renormiert.
 - **Tests:** `tests/test_router_inhibit_integration.py` (Inhibit senkt Auswahlwahrscheinlichkeit)
 - **Done wenn:** Tests + Status ok.

9. [DONE] **T22: End-to-End-Ping-Pong-Break**
 - Datei: `tests/test_governor_breaks_pingpong.py`  
   - Synthese-Setup: Inputs, die abwechselnd i↔j triggern (k=2 für Ko-Aktivation).  
   - Erwartung: Nach ≤L Schritten verschwindet Ping-Pong (Inhibition greift).
 - **Done wenn:** Test grün, Status ok. ✅

---

## Phase 3 – Gruppen/Cluster & Reports
10. [OPEN] **T30: Clustering**
  - Datei: `swarm_moi/clustering.py`  
    Funktion `cluster_from_C(C) -> labels` (einfach: spektral oder „schwellwert-basiert“; keine schweren Libs).
  - **Tests:** `tests/test_clustering_simple.py` (klare Blöcke → korrekte Labels)
  - **Done wenn:** Tests + Status ok.

11. [OPEN] **T31: Kurzreporter**
  - Datei: `swarm_moi/reporter.py`  
    `build_report(C, loop_pairs, labels)` → Dict mit:
    - Top-Gruppen (Größe, Vertreter), Top-Loop-Paare, Inhibitions-History-Stub
  - **Tests:** `tests/test_reporter_schema.py` (Schema/Schlüssel vorhanden, Werte sinnig)
  - **Done wenn:** Tests + Status ok.

12. [OPEN] **T32: Minimal-Console-Report**
  - Datei: `tools/mini_report.py` → ruft Reporter auf und druckt 10-Zeilen-Zusammenfassung.
  - **Tests:** `tests/test_mini_report_cli.py` (subprocess, Exit 0, erwartete Schlüsselwörter)
  - **Done wenn:** Script liefert kurze, stabile Ausgabe.

---

## Phase 4 – Trainings-Spielzeug & Reg-Losses
13. [OPEN] **T40: MoE-Layer Wrapper**
  - Datei: `swarm_moi/moe_layer.py`  
    Mini-API: `forward(x)` ruft Router→Experten→Combine; Experten als einfache MLPs.
  - **Tests:** `tests/test_moe_shapes.py` (Batched Inputs → erwartete Shapes)
  - **Done wenn:** Tests + Status ok.

14. [OPEN] **T41: Losses**
  - Datei: `swarm_moi/losses.py`  
    `exclusivity_loss(C)`, `smoothness_loss(p_t, p_tm1)`, Platzhalter `load_balance_loss(p)`.
  - **Tests:** `tests/test_losses_gradients.py` (kein NaN, Gradients vorhanden)
  - **Done wenn:** Tests + Status ok.

15. [OPEN] **T42: Toy-Training**
  - Datei: `examples/train_toy.py`  
    - Kleines MLP mit 1 MoE-Schicht; generischer Klassifikations-Toy-Datensatz.  
    - Zwei Läufe: **mit** und **ohne** Governor.  
    - Metriken: Varianz der Expertennutzung, Loop-Score, Accuracy.
  - **Tests:** `tests/test_train_toy_smoke.py` (kurzer Lauf, erwartet verbesserte Stabilität mit Governor)
  - **Done wenn:** Test grün, Report zeigt weniger Ping-Pong.

---

## Phase 5 – Policies erzwingen (KI an die Leine)
16. [OPEN] **T50: Plan-Format & Validator**
  - Datei: `plans/README.md` (Schema), `tools/validate_plan.py`
  - **Tests:** `tests/test_plan_validator.py` (gute/schlechte YAMLs)
  - **Done wenn:** Validator akzeptiert nur korrektes Schema.

17. [OPEN] **T51: Pre-Commit/CI-Hook**
  - Pre-Target: `tools/precheck.sh` ruft Enforcer + Plan-Validator + Status an (schneller Modus).
  - **Tests:** `tests/test_precheck_integration.py` (simulierter Commit → ok/fail)
  - **Done wenn:** Hook verhindert Regelverstöße lokal.

---

## Phase 6 – Stabilisierung & Doku
18. [OPEN] **T60: Diff-Coverage-Gate**
  - In `tools/status.sh`: `diff-cover --fail-under=85`.
  - **Tests:** `tests/test_diff_coverage_gate.py` (erzeugt gezielt niedrige Diff-Coverage → Fail)
  - **Done wenn:** Gate greift.

19. [OPEN] **T61: Readme Minimal**
  - Datei: `README.md` → 1-Seiten-Überblick (Start, Ziel, Skripte, Tests, Report usage).
  - **Tests:** `tests/test_readme_links.py` (alle referenzierten Dateien vorhanden)
  - **Done wenn:** Test grün.

20. [OPEN] **T62: Beispiel-Prompts für KI**
  - Datei: `PROMPTS.md` mit drei Blöcken:
    - „Plan zuerst“ (nur YAML erzeugen)
    - „Patch im Guard-Block + 1 Test“
    - „Nur Fehlertests lesen, minimal fixen“
  - **Tests:** `tests/test_prompts_presence.py` (Markdown enthält Pflicht-Sektionen)
  - **Done wenn:** Test grün.

---

## Anhänge

### A) Pflicht-Plan (Vorlage)
```yaml
# plans/TXX.yaml
task_id: "TXX"
intent: "Kurzbeschreibung"
files:
- path: "swarm_moi/<datei>.py"
  regions: ["AI_EDIT"]
tests_expected:
- "tests/<name>::<case>"
risk: "niedrig|mittel|hoch"
rollback_hint: "git revert <commit>"
notes: ""
B) KI-Prompt (Plan zuerst)
nginx
Copy code
Erzeuge NUR eine Plan-Datei 'plans/TXX.yaml' gemäß Vorlage A.
Keine Codeänderung. Falls Infos fehlen: abbrechen und Aufzählung fehlender Infos liefern.
C) KI-Prompt (Patch minimal)
r
Copy code
Ändere NUR die im Plan gelisteten Dateien und NUR zwischen Guard-Markern.
Max 200 Zeilen, max 3 Dateien. Füge mind. 1 neuen Unit-Test hinzu.
Nach Änderung: zeige nur Diff-Ausschnitte der Guards.
Abschlussbedingung (Projekt „Step Done“)
tools/status.sh meldet grün.

examples/train_toy.py zeigt: Mit Governor weniger Loop/Ping-Pong bei gleicher/verbesserter Accuracy.

tools/mini_report.py gibt Top-Gruppen & Top-Loop-Paare in ≤10 Zeilen aus.