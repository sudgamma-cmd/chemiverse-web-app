# ⚗️ ChemiVerse AI: Molecular Informatics & Reactivity Suite

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-green.svg)](https://www.rdkit.org/)
[![PubChem API](https://img.shields.io/badge/Database-PubChemPy-orange.svg)](https://pubchem.ncbi.nlm.nih.gov/)
[![WebGL 3D](https://img.shields.io/badge/3D%20Viewer-py3Dmol-red.svg)](https://3dmol.csb.pitt.edu/)
[![PDF Engine](https://img.shields.io/badge/Export-ReportLab-purple.svg)](https://www.reportlab.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, full-stack cheminformatics workbench built for students, researchers, and educators. **ChemiVerse AI** allows users to input multiple organic and inorganic chemical compounds, visualize 2D skeletal structures, interact with 3D molecular conformers, determine exact atomic hybridization states, simulate pairwise chemical reactivity against an interactive 118-element Periodic Table, and automatically compile an accredited multi-page laboratory PDF report.

---

## 🌟 Key Features

* **Multi-Compound Batch Processing:** Enter SMILES strings, chemical names (e.g., *Aspirin*, *Ethanol*), or inorganic chemical formulas (e.g., $\text{KMnO}_4$, $\text{FeCl}_3$, $\text{H}_2\text{SO}_4$) simultaneously.
* **Hybrid Organic & Inorganic Architecture:**
  * **Organics:** Parsed via **RDKit** for exact topological descriptors, aromaticity checks, rotatable bonds, and 3D Universal Force Field (UFF) energy minimization.
  * **Inorganics & Salts:** Automatically routed to **PubChemPy** for structural CIDs, coordinate fetching, and elemental composition checks.
* **Atomic Hybridization Calculator:** Automatically calculates non-hydrogen atomic orbital hybridization distributions ($\text{sp}$, $\text{sp}^2$, $\text{sp}^3$, $d^2\text{sp}^3$, tetrahedral, octahedral coordination geometry).
* **Dual-View Structural Visualization:**
  * **2D Vector Projections:** High-resolution 2D depictions rendered dynamically.
  * **3D Ball-and-Stick WebGL Canvas:** Interactive spatial models rendered using `py3Dmol` (full mouse rotation, panning, and zoom).
* **Interactive Periodic Table (118 Elements):**
  * **Elemental Lookup Mode:** View atomic number, electron configuration, category, and electronegativity.
  * **AI Reactivity Cross-Check:** Pair any currently loaded compound with an element tile to evaluate redox sensitivity, electrophilic attacks, displacement reactions, or chelation complexes.
* **Automated Laboratory PDF Report:** One-click compilation of a publication-grade multi-molecule PDF report complete with student name, registration ID, 2D structure figures, and tabular property summaries using **ReportLab**.

---

## 🏗️ System Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │              User Input Layer                │
                        │  (Student Name, Reg No, Multi-Compound List) │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │          Dual Cheminformatics Engine         │
                        ├──────────────────────┬───────────────────────┤
                        │    RDKit Pipeline    │   PubChemPy Routing   │
                        │   (Organic SMILES)   │  (Inorganic Formulas) │
                        └──────────┬───────────┴───────────┬───────────┘
                                   │                       │
                                   ▼                       ▼
                        ┌──────────────────────────────────────────────┐
                        │        Descriptors & Geometry Engine         │
                        │  - LogP, TPSA, MW, HBD/HBA, Csp3             │
                        │  - Orbital Hybridization (sp, sp², sp³)      │
                        │  - UFF 3D Conformer Coordinate Embedding    │
                        └──────────────────────┬───────────────────────┘
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
┌─────────────────────────┐                                           ┌─────────────────────────┐
│   Interactive Web App   │                                           │  Automated PDF Exporter │
│  - 2D Canvas & py3Dmol  │                                           │  - Student Metadata     │
│  - Periodic Table Grid  │                                           │  - 2D Renderings        │
│  - Reactivity Evaluator │                                           │  - Tabular Descriptors  │
└─────────────────────────┘                                           └─────────────────────────┘
```

---

## 📋 Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Reactive dashboard, sidebar inputs, tabbed navigation, session state. |
| **Organic Engine** | [RDKit](https://www.rdkit.org/) | Canonical SMILES, 2D rendering, 3D conformers, molecular descriptors. |
| **Inorganic Engine** | [PubChemPy](https://pubchempy.readthedocs.io/) | Resolving inorganic salts, coordination complexes, and mineral formulas. |
| **3D Visualization** | [py3Dmol](https://3dmol.csb.pitt.edu/) | WebGL hardware-accelerated ball-and-stick spatial inspection. |
| **Document Generator** | [ReportLab](https://www.reportlab.com/) | Generates print-ready multi-page student laboratory PDF reports. |
| **Data Handling** | [Pandas](https://pandas.pydata.org/) | Property matrices, tabular displays, and structured data filtering. |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.10 or 3.11 installed on your workstation.

### 2. Clone the Repository
```bash
git clone https://github.com/<YOUR-GITHUB-USERNAME>/chemiverse-web-app.git
cd chemiverse-web-app
```

### 3. Set Up a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run chemiverse_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📦 Repository Structure

```
├── .gitignore
├── README.md               # Project documentation and guide
├── requirements.txt        # Python dependency manifest
├── chemiverse_app.py       # Main Streamlit application and PDF pipeline
└── sample_reports/         # Sample generated laboratory reports (.pdf)
```

---

## 🧪 Sample Input Queries

You can test the application with mixed organic and inorganic entries:

```text
CC(=O)Oc1ccccc1C(=O)O    # Aspirin (Organic Carboxylic Ester)
KMnO4                    # Potassium Permanganate (Inorganic Oxidizer)
CCO                      # Ethanol (Primary Protic Alcohol)
FeCl3                    # Iron(III) Chloride (Lewis Acid / Coordination Precursor)
c1ccccc1                 # Benzene (Aromatic Hydrocarbon)
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍🔬 Author

* **Pallipati Sudhakar** – Undergraduate Student & Rail/Cheminformatics Enthusiast
* Institution: SRM Institute of Science and Technology
