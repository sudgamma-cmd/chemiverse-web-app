import streamlit as st
import os
import io
import tempfile
import json
from datetime import datetime

# Direct Imports without silent try-except
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem, Draw
import pubchempy as pcp
import py3Dmol
import streamlit.components.v1 as components
import pandas as pd

# PDF Generation (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChemiVerse - Molecular & Elemental Lab",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 12px; }
  .stTabs [data-baseweb="tab"] {
    background-color: #f1f5f9;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS: HYBRIDIZATION & PROPERTIES
# ---------------------------------------------------------
def get_hybridization_summary(mol):
    if not mol:
        return "N/A"
    hyb_counts = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        hyb = str(atom.GetHybridization()).replace("UNSPECIFIED", "Other")
        hyb_name = hyb.replace("HYBRIDIZATIONTYPE.", "")
        hyb_counts[hyb_name] = hyb_counts.get(hyb_name, 0) + 1
    
    parts = [f"{count} atom(s) {k}" for k, count in sorted(hyb_counts.items())]
    return ", ".join(parts) if parts else "No heavy atoms"

def analyze_organic_compound(smiles_or_name):
    mol = Chem.MolFromSmiles(smiles_or_name)
    if mol is None:
        try:
            results = pcp.get_compounds(smiles_or_name, 'name')
            if results:
                smiles = results[0].canonical_smiles
                mol = Chem.MolFromSmiles(smiles)
        except Exception:
            pass

    if mol is None:
        return None

    can_smiles = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    molar_refractivity = Crippen.MolMR(mol)
    hyb_summary = get_hybridization_summary(mol)

    mol_3d = Chem.AddHs(mol)
    has_3d = False
    try:
        res = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if res == 0:
            AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
            has_3d = True
    except Exception:
        has_3d = False

    return {
        "type": "Organic / Covalent",
        "input": smiles_or_name,
        "mol": mol,
        "mol_3d": mol_3d if has_3d else None,
        "canonical_smiles": can_smiles,
        "formula": formula,
        "mw": round(mw, 3),
        "logp": round(logp, 3),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rot_bonds": rot_bonds,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "aliphatic_rings": aliphatic_rings,
        "heavy_atoms": heavy_atoms,
        "fraction_csp3": round(fraction_csp3, 3),
        "molar_refractivity": round(molar_refractivity, 2),
        "hybridization": hyb_summary,
        "is_inorganic": False
    }

def analyze_inorganic_compound(formula_or_name):
    try:
        compounds = pcp.get_compounds(formula_or_name, 'name')
        if not compounds:
            compounds = pcp.get_compounds(formula_or_name, 'formula')
        if not compounds:
            return None
        
        comp = compounds[0]
        mol = None
        hyb_summary = "Inorganic Lattice / Coordination"
        if comp.canonical_smiles:
            mol = Chem.MolFromSmiles(comp.canonical_smiles)
            if mol:
                hyb_summary = get_hybridization_summary(mol)

        return {
            "type": "Inorganic / Ionic / Coordination",
            "input": formula_or_name,
            "mol": mol,
            "mol_3d": None,
            "canonical_smiles": comp.canonical_smiles or "N/A",
            "formula": comp.molecular_formula or formula_or_name,
            "mw": round(float(comp.molecular_weight or 0), 3),
            "logp": round(float(comp.xlogp or 0), 2) if comp.xlogp is not None else "N/A",
            "tpsa": round(float(comp.tpsa or 0), 2) if comp.tpsa is not None else "N/A",
            "hbd": comp.h_bond_donor_count or 0,
            "hba": comp.h_bond_acceptor_count or 0,
            "rot_bonds": comp.rotatable_bond_count or 0,
            "rings": "N/A",
            "aromatic_rings": "N/A",
            "aliphatic_rings": "N/A",
            "heavy_atoms": comp.heavy_atom_count or 0,
            "fraction_csp3": "N/A",
            "molar_refractivity": "N/A",
            "hybridization": hyb_summary,
            "charge": comp.charge or 0,
            "is_inorganic": True
        }
    except Exception:
        return None

def process_single_compound(text_input):
    text_input = text_input.strip()
    if not text_input:
        return None
    res = analyze_organic_compound(text_input)
    if res:
        return res
    res = analyze_inorganic_compound(text_input)
    if res:
        return res
    return None

def render_2d_molecule_image(mol, size=(350, 250)):
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

def render_3d_molecule_webgl(mol_3d):
    if mol_3d is None:
        st.info("3D Conformer not applicable or coordinates unoptimized.")
        return
    mblock = Chem.MolToMolBlock(mol_3d)
    viewer = py3Dmol.view(width=420, height=300)
    viewer.addModel(mblock, "mol")
    viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.28}})
    viewer.setBackgroundColor("#0f172a")
    viewer.zoomTo()
    components.html(viewer._make_html(), height=310)

def generate_pdf_report(user_name, user_reg, compounds_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15,
        leading=18, textColor=colors.HexColor("#0f172a"), alignment=1, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'SubTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9,
        leading=12, textColor=colors.HexColor("#2563eb"), alignment=1, spaceAfter=10
    )
    meta_style = ParagraphStyle(
        'Meta', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5,
        leading=11, textColor=colors.HexColor("#475569")
    )
    table_header_style = ParagraphStyle(
        'TH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8,
        leading=10, textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        'TD', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5,
        leading=10, textColor=colors.HexColor("#1e293b")
    )

    elements = []
    elements.append(Paragraph("CHEMINFORMATICS & REACTION LABORATORY REPORT", title_style))
    elements.append(Paragraph("AUTOMATED MULTI-COMPOUND STRUCTURAL & HYBRIDIZATION ANALYSIS", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

    meta_data = [
        [
            Paragraph(f"<b>Researcher / Student:</b> {user_name or 'Independent'}", meta_style),
            Paragraph(f"<b>Registration ID:</b> {user_reg or 'N/A'}", meta_style),
            Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %B %Y')}", meta_style),
            Paragraph(f"<b>Molecules Analyzed:</b> {len(compounds_data)}", meta_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[150, 120, 130, 120])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    for idx, c in enumerate(compounds_data, 1):
        elements.append(Paragraph(f"<b>Molecule {idx}: {c['formula']} ({c['type']})</b>", styles['Heading2']))
        elements.append(Paragraph(f"<b>Input Query:</b> <code>{c['input']}</code> | <b>Canonical SMILES:</b> <code>{c['canonical_smiles']}</code>", meta_style))
        elements.append(Spacer(1, 6))

        img_temp_path = None
        if c['mol'] is not None:
            try:
                img = render_2d_molecule_image(c['mol'], size=(320, 200))
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    img.save(tmp_file.name)
                    img_temp_path = tmp_file.name
            except Exception:
                img_temp_path = None

        prop_rows = [
            [Paragraph("Property", table_header_style), Paragraph("Calculated Value", table_header_style), Paragraph("Interpretation & Chemical Context", table_header_style)],
            [Paragraph("Molecular Formula", table_cell_style), Paragraph(str(c['formula']), table_cell_style), Paragraph("Elemental stoichiometry defining the compound.", table_cell_style)],
            [Paragraph("Molecular Weight", table_cell_style), Paragraph(f"{c['mw']} g/mol", table_cell_style), Paragraph("Average molar mass governing mass conservation & diffusion.", table_cell_style)],
            [Paragraph("Hybridization State", table_cell_style), Paragraph(str(c['hybridization']), table_cell_style), Paragraph("Orbital mixing geometry dictating bond angles (sp, sp², sp³).", table_cell_style)],
            [Paragraph("LogP (Lipophilicity)", table_cell_style), Paragraph(str(c['logp']), table_cell_style), Paragraph("Octanol-water partition: positive=lipophilic, negative=hydrophilic.", table_cell_style)],
            [Paragraph("TPSA (Polar Surface)", table_cell_style), Paragraph(f"{c['tpsa']} Å²", table_cell_style), Paragraph("Sum of polar heteroatom surfaces; correlates with membrane permeability.", table_cell_style)],
            [Paragraph("H-Bond Donors / Acceptors", table_cell_style), Paragraph(f"{c['hbd']} / {c['hba']}", table_cell_style), Paragraph("Capacity for hydrogen bonding interactions with protic solvents.", table_cell_style)],
            [Paragraph("Rotatable Bonds", table_cell_style), Paragraph(str(c['rot_bonds']), table_cell_style), Paragraph("Free dihedral rotations defining dynamic structural flexibility.", table_cell_style)],
            [Paragraph("Ring Count (Arom / Aliph)", table_cell_style), Paragraph(f"{c['rings']} ({c['aromatic_rings']}/{c['aliphatic_rings']})", table_cell_style), Paragraph("Cyclic stiffness and conjugated π-electron aromatic delocalization.", table_cell_style)],
            [Paragraph("Fraction Csp³", table_cell_style), Paragraph(str(c['fraction_csp3']), table_cell_style), Paragraph("Proportion of sp³ carbon atoms representing 3D spatial saturation.", table_cell_style)],
        ]

        prop_table = Table(prop_rows, colWidths=[130, 110, 280])
        prop_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))

        if img_temp_path and os.path.exists(img_temp_path):
            img_flowable = Image(img_temp_path, width=2.4*inch, height=1.5*inch)
            combo_table = Table([[img_flowable, prop_table]], colWidths=[180, 340])
            combo_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ]))
            elements.append(combo_table)
        else:
            elements.append(prop_table)

        elements.append(Spacer(1, 14))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.sidebar.title("🧪 Lab Configuration")
user_name = st.sidebar.text_input("Student / Researcher Name", value="Sudhakar")
user_reg = st.sidebar.text_input("Registration / Roll No.", value="SRM-2026-CH01")

st.sidebar.markdown("---")
st.sidebar.subheader("Multi-Compound Input")

DEFAULT_COMPOUNDS = """CC(=O)Oc1ccccc1C(=O)O
KMnO4
CCO
FeCl3
c1ccccc1"""

raw_inputs = st.sidebar.text_area(
    "Enter compounds (one per line):",
    value=DEFAULT_COMPOUNDS,
    help="Supports Organic SMILES, chemical names (Aspirin, Ethanol), and inorganic formulas (KMnO4, NaCl, H2SO4)."
)

compounds_list = [c.strip() for c in raw_inputs.split("\n") if c.strip()]

st.title("🔬 ChemiVerse AI: Molecular & Elemental Workbench")

parsed_data = []
for c_input in compounds_list:
    res = process_single_compound(c_input)
    if res:
        parsed_data.append(res)

if not parsed_data:
    st.warning("Please enter at least one valid organic SMILES or inorganic formula in the sidebar.")
    st.stop()

tab_overview, tab_details, tab_periodic, tab_export = st.tabs([
    "📊 Structures (2D & 3D)",
    "🔬 Hybridization & Descriptors",
    "⚛️ Interactive Periodic Table",
    "📄 PDF Report Generation"
])

with tab_overview:
    st.subheader(f"Batch Analysis ({len(parsed_data)} Compounds)")
    for idx, c in enumerate(parsed_data):
        with st.expander(f"Compound {idx+1}: {c['formula']} ({c['type']}) — {c['input']}", expanded=True):
            col2d, col3d, col_summary = st.columns([1, 1.1, 1.2])
            with col2d:
                st.markdown("**2D Skeletal Projection**")
                if c['mol'] is not None:
                    st.image(render_2d_molecule_image(c['mol']), use_column_width=True)
                else:
                    st.info("2D diagram not available for complex inorganic coordinate/salt.")
            with col3d:
                st.markdown("**3D Ball-and-Stick Conformer (WebGL)**")
                if c['mol_3d'] is not None:
                    render_3d_molecule_webgl(c['mol_3d'])
                else:
                    st.caption("3D interactive conformer requires RDKit coordinate embedding.")
            with col_summary:
                st.markdown("**Core Characteristics**")
                st.write(f"• **Formula:** `{c['formula']}`")
                st.write(f"• **Molecular Weight:** `{c['mw']} g/mol`")
                st.write(f"• **Hybridization:** `{c['hybridization']}`")
                st.write(f"• **LogP (Hydrophobicity):** `{c['logp']}`")
                st.write(f"• **Polar Surface (TPSA):** `{c['tpsa']} Å²`")
                st.write(f"• **Rotatable Bonds:** `{c['rot_bonds']}`")

with tab_details:
    st.subheader("Orbital Hybridization & Physicochemical Properties")
    summary_table = []
    for c in parsed_data:
        summary_table.append({
            "Input": c['input'],
            "Formula": c['formula'],
            "MW (g/mol)": c['mw'],
            "Hybridization Profile": c['hybridization'],
            "LogP": c['logp'],
            "TPSA (Å²)": c['tpsa'],
            "H-Donors": c['hbd'],
            "H-Acceptors": c['hba'],
            "Rotatable Bonds": c['rot_bonds'],
            "Csp3 Fraction": c['fraction_csp3']
        })
    st.dataframe(pd.DataFrame(summary_table), use_container_width=True)

# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 3: INTERACTIVE PERIODIC TABLE & REACTIVITY STUDIO
# ---------------------------------------------------------
with tab_periodic:
    st.subheader("⚛️ Interactive Periodic Table & Reactivity Studio")
    st.caption("Explore chemical elements, orbital configurations, and simulate pairwise reactivity with your batch!")

    ELEMENTS_DATA = [
        {"z": 1, "sym": "H", "name": "Hydrogen", "mass": 1.008, "cat": "Nonmetal", "color": "#1e3a8a", "config": "1s¹", "en": 2.20},
        {"z": 2, "sym": "He", "name": "Helium", "mass": 4.002, "cat": "Noble Gas", "color": "#581c87", "config": "1s²", "en": "N/A"},
        {"z": 3, "sym": "Li", "name": "Lithium", "mass": 6.94, "cat": "Alkali Metal", "color": "#991b1b", "config": "[He] 2s¹", "en": 0.98},
        {"z": 4, "sym": "Be", "name": "Beryllium", "mass": 9.012, "cat": "Alkaline Earth", "color": "#9a3412", "config": "[He] 2s²", "en": 1.57},
        {"z": 5, "sym": "B", "name": "Boron", "mass": 10.81, "cat": "Metalloid", "color": "#115e59", "config": "[He] 2s² 2p¹", "en": 2.04},
        {"z": 6, "sym": "C", "name": "Carbon", "mass": 12.011, "cat": "Nonmetal", "color": "#1e3a8a", "config": "[He] 2s² 2p²", "en": 2.55},
        {"z": 7, "sym": "N", "name": "Nitrogen", "mass": 14.007, "cat": "Nonmetal", "color": "#1e3a8a", "config": "[He] 2s² 2p³", "en": 3.04},
        {"z": 8, "sym": "O", "name": "Oxygen", "mass": 15.999, "cat": "Nonmetal", "color": "#1e3a8a", "config": "[He] 2s² 2p⁴", "en": 3.44},
        {"z": 9, "sym": "F", "name": "Fluorine", "mass": 18.998, "cat": "Halogen", "color": "#065f46", "config": "[He] 2s² 2p⁵", "en": 3.98},
        {"z": 10, "sym": "Ne", "name": "Neon", "mass": 20.180, "cat": "Noble Gas", "color": "#581c87", "config": "[He] 2s² 2p⁶", "en": "N/A"},
        {"z": 11, "sym": "Na", "name": "Sodium", "mass": 22.990, "cat": "Alkali Metal", "color": "#991b1b", "config": "[Ne] 3s¹", "en": 0.93},
        {"z": 12, "sym": "Mg", "name": "Magnesium", "mass": 24.305, "cat": "Alkaline Earth", "color": "#9a3412", "config": "[Ne] 3s²", "en": 1.31},
        {"z": 13, "sym": "Al", "name": "Aluminum", "mass": 26.982, "cat": "Post-Transition", "color": "#475569", "config": "[Ne] 3s² 3p¹", "en": 1.61},
        {"z": 14, "sym": "Si", "name": "Silicon", "mass": 28.085, "cat": "Metalloid", "color": "#115e59", "config": "[Ne] 3s² 3p²", "en": 1.90},
        {"z": 15, "sym": "P", "name": "Phosphorus", "mass": 30.974, "cat": "Nonmetal", "color": "#1e3a8a", "config": "[Ne] 3s² 3p³", "en": 2.19},
        {"z": 16, "sym": "S", "name": "Sulfur", "mass": 32.06, "cat": "Nonmetal", "color": "#1e3a8a", "config": "[Ne] 3s² 3p⁴", "en": 2.58},
        {"z": 17, "sym": "Cl", "name": "Chlorine", "mass": 35.45, "cat": "Halogen", "color": "#065f46", "config": "[Ne] 3s² 3p⁵", "en": 3.16},
        {"z": 18, "sym": "Ar", "name": "Argon", "mass": 39.948, "cat": "Noble Gas", "color": "#581c87", "config": "[Ne] 3s² 3p⁶", "en": "N/A"},
        {"z": 19, "sym": "K", "name": "Potassium", "mass": 39.098, "cat": "Alkali Metal", "color": "#991b1b", "config": "[Ar] 4s¹", "en": 0.82},
        {"z": 20, "sym": "Ca", "name": "Calcium", "mass": 40.078, "cat": "Alkaline Earth", "color": "#9a3412", "config": "[Ar] 4s²", "en": 1.00},
        {"z": 26, "sym": "Fe", "name": "Iron", "mass": 55.845, "cat": "Transition Metal", "color": "#334155", "config": "[Ar] 3d⁶ 4s²", "en": 1.83},
        {"z": 29, "sym": "Cu", "name": "Copper", "mass": 63.546, "cat": "Transition Metal", "color": "#334155", "config": "[Ar] 3d¹⁰ 4s¹", "en": 1.90},
        {"z": 30, "sym": "Zn", "name": "Zinc", "mass": 65.38, "cat": "Transition Metal", "color": "#334155", "config": "[Ar] 3d¹⁰ 4s²", "en": 1.65},
        {"z": 35, "sym": "Br", "name": "Bromine", "mass": 79.904, "cat": "Halogen", "color": "#065f46", "config": "[Ar] 3d¹⁰ 4s² 4p⁵", "en": 2.96},
        {"z": 47, "sym": "Ag", "name": "Silver", "mass": 107.87, "cat": "Transition Metal", "color": "#334155", "config": "[Kr] 4d¹⁰ 5s¹", "en": 1.93},
        {"z": 53, "sym": "I", "name": "Iodine", "mass": 126.90, "cat": "Halogen", "color": "#065f46", "config": "[Kr] 4d¹⁰ 5s² 5p⁵", "en": 2.66},
        {"z": 79, "sym": "Au", "name": "Gold", "mass": 196.97, "cat": "Transition Metal", "color": "#334155", "config": "[Xe] 4f¹⁴ 5d¹⁰ 6s¹", "en": 2.54},
        {"z": 80, "sym": "Hg", "name": "Mercury", "mass": 200.59, "cat": "Transition Metal", "color": "#334155", "config": "[Xe] 4f¹⁴ 5d¹⁰ 6s²", "en": 2.00},
    ]

    # Category Filter Pills
    categories = ["All", "Alkali Metal", "Alkaline Earth", "Transition Metal", "Metalloid", "Nonmetal", "Halogen", "Noble Gas"]
    selected_cat = st.segmented_control("Filter by Chemical Family:", categories, default="All") if hasattr(st, "segmented_control") else st.radio("Filter by Family:", categories, horizontal=True)

    filtered_elements = [el for el in ELEMENTS_DATA if selected_cat == "All" or el["cat"] == selected_cat]

    # Render Clean Touch-Friendly Responsive Grid (Works smoothly on both Mobile Safari and Desktop)
    cols = st.columns(6)
    if "active_elem" not in st.session_state:
        st.session_state["active_elem"] = ELEMENTS_DATA[10] # Default Na

    for idx, el in enumerate(filtered_elements):
        c_idx = idx % 6
        with cols[c_idx]:
            btn_label = f"{el['sym']} ({el['z']})\n{el['name']}"
            if st.button(btn_label, key=f"pel_{el['sym']}", use_container_width=True):
                st.session_state["active_elem"] = el

    elem = st.session_state["active_elem"]

    # Active Element Card
    st.markdown("---")
    c_info1, c_info2 = st.columns([1, 1.5])
    with c_info1:
        st.markdown(f"""
        <div style="background:{elem['color']}; color:white; padding:18px; border-radius:12px; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
          <div style="font-size:14px; opacity:0.85;">Atomic Number: {elem['z']}</div>
          <div style="font-size:42px; font-weight:800; line-height:1.1;">{elem['sym']}</div>
          <div style="font-size:18px; font-weight:600;">{elem['name']}</div>
          <div style="font-size:13px; margin-top:4px; opacity:0.9;">Weight: {elem['mass']} u</div>
        </div>
        """, unsafe_allow_html=True)
    with c_info2:
        st.markdown("#### 🔬 Elemental Properties")
        st.write(f"• **Chemical Family:** `{elem['cat']}`")
        st.write(f"• **Electron Configuration:** `{elem['config']}`")
        st.write(f"• **Pauling Electronegativity:** `{elem['en']}`")

    # Pairwise Reactivity Section
    st.markdown("### ⚡ Test Reactivity Against Your Compound Batch")
    target_compound = st.selectbox(
        "Select Molecule to React with:",
        options=[f"{c['formula']} ({c['input']})" for c in parsed_data],
        key="target_rxn_mol"
    )

    sym = elem['sym']
    st.markdown(f"""
    <div style="background: #0f172a; color: white; padding: 16px 20px; border-radius: 10px; border-left: 5px solid #2563eb; margin-top: 8px;">
      <h4 style="margin: 0 0 8px 0; color: #60a5fa;">Reactivity Assessment: {target_compound} + Elemental {elem['name']} ({sym})</h4>
    """, unsafe_allow_html=True)

    if sym in ['Na', 'Li', 'K']:
        st.write("• **Reaction Class:** Strong Single-Displacement Redox / Acid-Base Gas Evolution.")
        st.write(f"• **Mechanism:** Elemental **{elem['name']}** readily donates its single valence electron (E° = -2.71 V) to reduce acidic protons in carboxyl (-COOH) or hydroxyl (-OH) groups, forming hydrogen gas (H₂↑) and the corresponding ionic metal salt.")
        st.write("• **Thermodynamics:** Highly exothermic (ΔG < 0); requires anhydrous conditions to prevent atmospheric moisture ignition.")
    elif sym in ['Cl', 'Br', 'F', 'I']:
        st.write("• **Reaction Class:** Electrophilic Addition / Free-Radical Halogenation.")
        st.write(f"• **Mechanism:** Elemental **{elem['name']}** acts as a powerful electrophile, halogenating alkene double bonds or substituting activated aromatic ortho/para positions in the presence of a Lewis acid catalyst.")
        st.write("• **Thermodynamics:** Spontaneous under UV photo-activation; produces hydrogen halide (HX) gas by-products.")
    elif sym in ['Fe', 'Cu', 'Zn', 'Mn']:
        st.write("• **Reaction Class:** Transition Metal Coordination / Chelation Complexation.")
        st.write(f"• **Mechanism:** **{elem['name']}** provides vacant d-orbitals to accept electron pairs from Lewis basic donors (oxygen lone pairs in esters/acids, nitrogen lone pairs in amines), forming stable 4-coordinate or 6-coordinate octahedral chelate complexes.")
        st.write("• **Thermodynamics:** Coordinate bond formation stabilizes the system; alters color, solubility, and redox potentials.")
    elif sym in ['He', 'Ne', 'Ar']:
        st.write("• **Reaction Class:** Chemical Inertness / Passive Shielding.")
        st.write(f"• **Mechanism:** **{elem['name']}** possesses a complete octet valence shell (s²p⁶) with extremely high ionization potential. It exhibits zero reactivity toward the compound and serves as an ideal inert atmosphere.")
    else:
        st.write("• **Reaction Class:** Heterogeneous Interfacial Adsorption / Surface Catalysis.")
        st.write(f"• **Mechanism:** Under ambient thermal conditions, elemental **{elem['name']}** remains kinetically passive toward the compound. At elevated temperatures or over activated metallic surfaces, it facilitates catalytic bond cleavage.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab_export:
    st.subheader("Download Accredited Laboratory Report")
    st.write("Generate a publication-grade multi-molecule PDF report matching the laboratory format.")
    
    if st.button("🚀 Compile & Download Laboratory PDF", type="primary"):
        with st.spinner("Generating PDF Document with ReportLab..."):
            pdf_bytes = generate_pdf_report(user_name, user_reg, parsed_data)
            st.download_button(
                label="📥 Download Cheminformatics_Multi_Molecule_Report.pdf",
                data=pdf_bytes,
                file_name=f"Cheminformatics_Report_{user_name}.pdf",
                mime="application/pdf"
            )
        st.success("Report successfully generated!")
