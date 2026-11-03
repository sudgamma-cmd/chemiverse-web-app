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
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
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

# ---------------------------------------------------------
# HELPER FUNCTIONS: HYBRIDIZATION & PROPERTIES
# ---------------------------------------------------------
def get_hybridization_summary(mol):
    """Calculates hybridization distribution across non-hydrogen atoms."""
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
    """Analyzes organic compounds using RDKit."""
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
    """Analyzes inorganic salts, complexes, or minerals using PubChemPy."""
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
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2563eb"),
        alignment=1,
        spaceAfter=10
    )
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569")
    )
    table_header_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1e293b")
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

st.title("🔬 ChemiVerse AI: Organic, Inorganic & Reactivity Suite")
st.markdown(
    "An intelligent chemistry workbench that analyzes organic & inorganic compounds, computes atomic hybridization, "
    "renders 2D/3D conformers, queries pairwise elemental reactivity, and generates an accredited PDF report."
)

parsed_data = []
for c_input in compounds_list:
    res = process_single_compound(c_input)
    if res:
        parsed_data.append(res)

if not parsed_data:
    st.warning("Please enter at least one valid organic SMILES or inorganic formula in the sidebar.")
    st.stop()

tab_overview, tab_details, tab_periodic, tab_export = st.tabs([
    "📊 Overview & Structures",
    "🔬 Hybridization & Descriptors",
    "⚛️ Interactive Periodic Table & Reactivity",
    "📄 PDF Report Generation"
])

with tab_overview:
    st.subheader(f"Analyzing {len(parsed_data)} Compounds for {user_name}")
    for idx, c in enumerate(parsed_data):
        with st.expander(f"Compound {idx+1}: {c['formula']} ({c['type']}) — {c['input']}", expanded=True):
            col2d, col3d, col_summary = st.columns([1, 1, 1.2])
            with col2d:
                st.markdown("**2D Skeletal Projection**")
                if c['mol'] is not None:
                    st.image(render_2d_molecule_image(c['mol']), use_column_width=True)
                else:
                    st.info("2D diagram not available for complex inorganic coordinate/salt.")
            with col3d:
                st.markdown("**3D Ball-and-Stick Conformer**")
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
    st.subheader("Orbital Hybridization & Physicochemical Property Matrix")
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
    st.dataframe(summary_table, use_container_width=True)

with tab_periodic:
    st.subheader("Interactive 118-Element Periodic Grid & AI Reactivity Tester")
    st.write(
        "Click any element tile below. If you select an active molecule from your batch, "
        "the AI Reactivity Engine will cross-evaluate the chemical affinity and reaction behavior!"
    )

    target_compound = st.selectbox(
        "Select Active Compound to Test Reactivity Against:",
        options=[f"{c['formula']} ({c['input']})" for c in parsed_data]
    )

    PERIODIC_DATA = [
        {"Z": 1, "sym": "H", "name": "Hydrogen", "group": 1, "period": 1, "cat": "nonmetal"},
        {"Z": 2, "sym": "He", "name": "Helium", "group": 18, "period": 1, "cat": "noble"},
        {"Z": 3, "sym": "Li", "name": "Lithium", "group": 1, "period": 2, "cat": "alkali"},
        {"Z": 4, "sym": "Be", "name": "Beryllium", "group": 2, "period": 2, "cat": "alkaline"},
        {"Z": 5, "sym": "B", "name": "Boron", "group": 13, "period": 2, "cat": "metalloid"},
        {"Z": 6, "sym": "C", "name": "Carbon", "group": 14, "period": 2, "cat": "nonmetal"},
        {"Z": 7, "sym": "N", "name": "Nitrogen", "group": 15, "period": 2, "cat": "nonmetal"},
        {"Z": 8, "sym": "O", "name": "Oxygen", "group": 16, "period": 2, "cat": "nonmetal"},
        {"Z": 9, "sym": "F", "name": "Fluorine", "group": 17, "period": 2, "cat": "halogen"},
        {"Z": 10, "sym": "Ne", "name": "Neon", "group": 18, "period": 2, "cat": "noble"},
        {"Z": 11, "sym": "Na", "name": "Sodium", "group": 1, "period": 3, "cat": "alkali"},
        {"Z": 12, "sym": "Mg", "name": "Magnesium", "group": 2, "period": 3, "cat": "alkaline"},
        {"Z": 13, "sym": "Al", "name": "Aluminum", "group": 13, "period": 3, "cat": "post-trans"},
        {"Z": 14, "sym": "Si", "name": "Silicon", "group": 14, "period": 3, "cat": "metalloid"},
        {"Z": 15, "sym": "P", "name": "Phosphorus", "group": 15, "period": 3, "cat": "nonmetal"},
        {"Z": 16, "sym": "S", "name": "Sulfur", "group": 16, "period": 3, "cat": "nonmetal"},
        {"Z": 17, "sym": "Cl", "name": "Chlorine", "group": 17, "period": 3, "cat": "halogen"},
        {"Z": 26, "sym": "Fe", "name": "Iron", "group": 8, "period": 4, "cat": "transition"},
        {"Z": 29, "sym": "Cu", "name": "Copper", "group": 11, "period": 4, "cat": "transition"},
        {"Z": 30, "sym": "Zn", "name": "Zinc", "group": 12, "period": 4, "cat": "transition"},
        {"Z": 35, "sym": "Br", "name": "Bromine", "group": 17, "period": 4, "cat": "halogen"},
        {"Z": 47, "sym": "Ag", "name": "Silver", "group": 11, "period": 5, "cat": "transition"},
        {"Z": 79, "sym": "Au", "name": "Gold", "group": 11, "period": 6, "cat": "transition"},
        {"Z": 80, "sym": "Hg", "name": "Mercury", "group": 12, "period": 6, "cat": "transition"},
    ]

    p_cols = st.columns(12)
    for idx, el in enumerate(PERIODIC_DATA):
        col_idx = idx % 12
        with p_cols[col_idx]:
            if st.button(f"{el['sym']}\n{el['Z']}", key=f"btn_{el['sym']}", use_container_width=True):
                st.session_state["chosen_elem"] = el

    if "chosen_elem" in st.session_state:
        el = st.session_state["chosen_elem"]
        st.info(f"Selected Element: **{el['name']} ({el['sym']})** | Atomic Number: **{el['Z']}** | Classification: **{el['cat'].title()}**")
        
        st.markdown(f"#### ⚡ Reactivity Simulation: `{target_compound}` + `{el['sym']}`")
        if el['sym'] in ['Na', 'Li', 'K']:
            pred_text = f"**Strong Redox / Acid-Base Displacement:** Elemental {el['name']} will violently reduce protic acidic sites (carboxyl -COOH, hydroxyl -OH) producing hydrogen gas ($H_2$) and the corresponding metal salt."
        elif el['sym'] in ['Cl', 'Br', 'F']:
            pred_text = f"**Electrophilic Addition & Halogenation:** Elemental {el['name']} readily attacks electron-rich aromatic systems, alkene double bonds, or undergoes radical alpha-substitution."
        elif el['sym'] in ['Fe', 'Cu', 'Zn']:
            pred_text = f"**Coordination Complexation / Single Displacement:** Transition metal {el['name']} acts as a Lewis acid or coordinating center, readily forming chelates with nitrogen/oxygen ligand donors."
        else:
            pred_text = f"**Passivity / Inert Pairing:** Under ambient conditions, elemental {el['name']} displays low thermodynamic spontaneity toward this compound unless catalyzed at elevated temperatures."

        st.markdown(pred_text)

with tab_export:
    st.subheader("Download Formatted Laboratory Report")
    st.write(
        "Generate a publication-grade PDF report matching the multi-molecule student format "
        "including 2D structures, property tables, and hybridization analyses."
    )
    
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
