from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import math

app = FastAPI(title="Material Discovery Engine v2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SmilesRequest(BaseModel):
    smiles: str
    candidate_name: Optional[str] = ""


class BatchRequest(BaseModel):
    candidates: List[SmilesRequest]


def compute_descriptors(smiles: str):
    """Compute molecular descriptors using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem
        from rdkit.Chem import rdMolDescriptors as rdmd

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = rdmd.CalcNumHBD(mol)
        hba = rdmd.CalcNumHBA(mol)
        rotatable = rdmd.CalcNumRotatableBonds(mol)
        rings = rdmd.CalcNumRings(mol)
        arom_rings = rdmd.CalcNumAromaticRings(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        fsp3 = rdmd.CalcFractionCSP3(mol)
        try:
            qed = QED.qed(mol)
        except Exception:
            qed = 0.0

        # Lipinski rule of 5 pass/fail
        lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

        # Synthetic accessibility heuristic (0-10, lower = easier)
        sa_score = max(1.0, min(10.0, 1.0 + (heavy_atoms / 20.0) + (rings * 0.5) + (arom_rings * 0.3)))

        # Environmental score: prefer lower logp and tpsa
        env_score = max(0.0, min(10.0, 10.0 - abs(logp) - (tpsa / 50.0)))

        # Overall material score (normalized)
        mat_score = round(
            (qed * 40) +
            (min(fsp3, 1.0) * 20) +
            ((1.0 - min(sa_score / 10.0, 1.0)) * 20) +
            (min(env_score / 10.0, 1.0) * 20),
            2,
        )

        return {
            "mw": round(mw, 3),
            "logp": round(logp, 3),
            "tpsa": round(tpsa, 3),
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rotatable,
            "ring_count": rings,
            "aromatic_rings": arom_rings,
            "heavy_atoms": heavy_atoms,
            "fsp3": round(fsp3, 4),
            "qed": round(qed, 4),
            "lipinski_pass": lipinski,
            "sa_score": round(sa_score, 2),
            "env_score": round(env_score, 2),
            "material_score": mat_score,
        }, None

    except ImportError:
        # Fallback if RDKit not installed: rough estimates from atom counting
        atoms = len([c for c in smiles if c.isalpha() and c.upper() == c and c != 'H'])
        mw_est = atoms * 12.0 + smiles.count('O') * 4.0 + smiles.count('N') * 2.0
        logp_est = round(smiles.count('C') * 0.5 - smiles.count('O') * 0.3, 2)
        return {
            "mw": round(max(mw_est, 18.0), 2),
            "logp": logp_est,
            "tpsa": round(smiles.count('O') * 20.0 + smiles.count('N') * 26.0, 2),
            "hbd": smiles.count('O') + smiles.count('N'),
            "hba": smiles.count('O') + smiles.count('N'),
            "rotatable_bonds": max(0, len(smiles) // 5 - 2),
            "ring_count": smiles.count('1') // 2,
            "aromatic_rings": smiles.count('c') // 6,
            "heavy_atoms": atoms,
            "fsp3": 0.5,
            "qed": 0.5,
            "lipinski_pass": True,
            "sa_score": 3.0,
            "env_score": 7.0,
            "material_score": 55.0,
            "note": "RDKit not installed – estimated values",
        }, None


@app.get("/")
def root():
    return {"ok": True, "message": "Material Discovery Engine v2.5 RDKit API"}


@app.post("/analyze_smiles")
def analyze_smiles(req: SmilesRequest):
    result, err = compute_descriptors(req.smiles)
    if err:
        return {"ok": False, "error": err}
    return {"ok": True, "smiles": req.smiles, "name": req.candidate_name, "descriptors": result}


@app.post("/batch_analyze")
def batch_analyze(req: BatchRequest):
    results = []
    for item in req.candidates:
        desc, err = compute_descriptors(item.smiles)
        if err:
            results.append({"name": item.candidate_name, "smiles": item.smiles, "ok": False, "error": err})
        else:
            results.append({"name": item.candidate_name, "smiles": item.smiles, "ok": True, "descriptors": desc})
    # Sort by material_score descending
    results.sort(key=lambda x: x.get("descriptors", {}).get("material_score", 0), reverse=True)
    return {"ok": True, "results": results}


@app.get("/health")
def health():
    try:
        from rdkit import Chem
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False
    return {"ok": True, "rdkit": rdkit_ok}
