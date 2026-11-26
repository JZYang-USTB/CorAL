import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DESCRIPTORS = (
    "logP", "TPSA", "NumRotatableBonds", "HBA", "HBD", "MaxPartialCharge",
    "AvgIpc", "BertzCT", "LabuteASA", "HallKierAlpha", "InertialShapeFactor",
    "Eccentricity", "NPR1", "RadiusOfGyration", "PBF",
)


def read_catalog(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def scaled_descriptor_matrix(rows):
    x = np.array([[float(row[name]) if row[name].strip() else np.nan
                   for name in DESCRIPTORS] for row in rows])
    if not np.isfinite(x).any(axis=0).all() or np.isinf(x).any():
        raise ValueError("Every descriptor column needs a finite value.")
    return StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(x))


def descriptor_contributions(rows, group):
    organic = [row for row in rows if int(row["group"]) != 11]
    target = np.array([int(row["group"]) == group for row in organic], dtype=bool)
    if not target.any() or target.all():
        raise ValueError("PCA needs a populated organic group and a populated comparison set.")
    pca = PCA(n_components=2, svd_solver="full")
    scores = pca.fit_transform(scaled_descriptor_matrix(organic))
    differences = scores[target].mean(axis=0) - scores[~target].mean(axis=0)
    contributions = np.abs(pca.components_ * differences[:, None]).sum(axis=0)
    ranking = sorted(zip(DESCRIPTORS, contributions.tolist()), key=lambda pair: -pair[1])
    return {
        "group": group,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "target_minus_remaining_mean": differences.tolist(),
        "descriptor_contributions": dict(ranking),
        "loadings": dict(zip(DESCRIPTORS, pca.components_.T.tolist(), strict=True)),
        "scores": {row["molecule_id"]: point.tolist()
                   for row, point in zip(organic, scores, strict=True)},
    }


def compute_descriptors(smiles, requested, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Descriptors3D

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not requested or set(requested) - set(DESCRIPTORS):
        raise ValueError("Supply valid SMILES and supported descriptor names.")
    functions_3d = {name: getattr(Descriptors3D, name) for name in DESCRIPTORS if hasattr(Descriptors3D, name)}
    if set(requested) & functions_3d.keys():
        molecule = Chem.AddHs(molecule)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(molecule, params) == -1:
            raise ValueError("RDKit could not generate a conformer for this SMILES.")
        molecule = Chem.RemoveHs(molecule)
    values = {}
    for name in requested:
        function = functions_3d.get(name) or getattr(Descriptors, "MolLogP" if name == "logP" else name)
        value = float(function(molecule))
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite descriptor {name} for the supplied SMILES.")
        values[name] = value
    return values
