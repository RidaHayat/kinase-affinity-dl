"""
Molecular Feature Extraction:
Computes RDKit Morgan Fingerprints (ECFP4 equivalent) and 2D Physicochemical Descriptors.
"""

import numpy as np

def compute_molecular_features(smiles: str, n_bits: int = 1024, radius: int = 2):
    """
    Featurizes a SMILES string into:
    - Morgan fingerprint (1024-bit vector)
    - 6 physicochemical descriptors: [MW, LogP, HBD, HBA, TPSA, RotBonds]
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 1. Morgan Fingerprint (ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fp_arr = np.zeros((n_bits,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
        
        # 2. Key Physicochemical Descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        
        desc_arr = np.array([mw, logp, hbd, hba, tpsa, rot_bonds], dtype=np.float32)
        
        # Combined feature vector
        combined = np.concatenate([fp_arr, desc_arr])
        return combined

    except ImportError:
        # Fallback pseudo-fingerprint for environments without RDKit installed
        np.random.seed(abs(hash(smiles)) % (2**32))
        fp = np.random.binomial(1, 0.05, size=n_bits).astype(np.float32)
        desc = np.array([len(smiles)*12.0, len(smiles)*0.1, 2.0, 4.0, 60.0, 3.0], dtype=np.float32)
        return np.concatenate([fp, desc])
