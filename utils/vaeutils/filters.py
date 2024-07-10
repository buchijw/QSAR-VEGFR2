import rdkit
import rdkit.Chem as Chem
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

class MOSES_Filters:
    def __init__(self, mcf_path:str, pains_path:str):
        _mcf = pd.read_csv(mcf_path)
        _pains = pd.read_csv(pains_path,
                            names=['smarts', 'names'])
        self.mcf_filters = [Chem.MolFromSmarts(x) for x in
                    _mcf['smarts'].values]
        self.pains_filters = [Chem.MolFromSmarts(x) for x in
                    _pains['smarts'].values]
        print('Filters loaded')
    def mol_passes_filters(self, mol,
                        allowed=None,
                        isomericSmiles=False):
        """
        Checks if mol
        * passes MCF and PAINS filters,
        * has only allowed atoms
        * is not charged
        """
        allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
        if mol is None:
            return 'NoMol'
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() != 0 and any(
                len(x) >= 8 for x in ring_info.AtomRings()
        ):
            return 'ManyRings'
        h_mol = Chem.AddHs(mol)
        if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
            return 'Charged'
        if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
            return 'AtomNotAllowed'
        if any(h_mol.HasSubstructMatch(smarts) for smarts in self.mcf_filters):
            return 'MCF'
        if any(h_mol.HasSubstructMatch(smarts) for smarts in self.pains_filters):
            return 'PAINS'
        smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
        if smiles is None or len(smiles) == 0:
            return 'Isomeric'
        if Chem.MolFromSmiles(smiles) is None:
            return 'Isomeric'
        return 'YES'
    
    def smiles_passes_filters(self, smiles,
                        allowed=None,
                        isomericSmiles=False):
        mol = Chem.MolFromSmiles(smiles)
        return self.mol_passes_filters(mol, allowed, isomericSmiles)
    
    def df_check_passes_filters(self, data:pd.DataFrame, smiles_col:str, allowed=None, isomericSmiles=False, drop=False):
        print('--- DATAFRAME MOSES FILTERS CHECK ---')
        temp_df = data.copy()
        print('Data points: %d\n'%(temp_df.shape[0]))
        print('Checking...')
        temp_df['MOSES_passed'] = temp_df[smiles_col].progress_apply(lambda x: self.smiles_passes_filters(x, allowed, isomericSmiles))
        print('Checked.\n')
        if drop:
            print('Filtering...')
            temp_df = temp_df[temp_df['MOSES_passed'] == 'YES']
            temp_df.reset_index(drop=True, inplace = True)
            print('Final data points: %d'%(temp_df.shape[0]))
        return temp_df