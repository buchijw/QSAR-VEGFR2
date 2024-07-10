from typing import Literal
import warnings
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
from sklearn.model_selection._split import _UnsupportedGroupCVMixin, _RepeatedSplits, _BaseKFold, GroupsConsumerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples, check_array, column_or_1d
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from collections import defaultdict


def scaffold_split(data: pd.DataFrame, smiles_col: str, test_size: float = 0.2, random_state: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a molecule dataset into training and test sets based on scaffolds.

    Parameters:
    - data (pd.DataFrame): The dataset containing molecule data.
    - smiles_col (str): The name of the column containing SMILES strings.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random state for reproducibility.

    Returns:
    - data_train (pd.DataFrame): Training set.
    - data_test (pd.DataFrame): Test set.
    """
    scaffolds = {}
    for idx, row in data.iterrows():
        smiles = row[smiles_col]
        # print(smiles)
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

    scaffold_lists = list(scaffolds.values())
    np.random.seed(random_state)
    np.random.shuffle(scaffold_lists)

    num_molecules = len(data)
    num_test = int(np.floor(test_size * num_molecules))
    train_idx, test_idx = [], []
    for scaffold_list in scaffold_lists:
        if len(test_idx) + len(scaffold_list) <= num_test:
            test_idx.extend(scaffold_list)
        else:
            train_idx.extend(scaffold_list)

    data_train = data.iloc[train_idx]
    data_test = data.iloc[test_idx]

    return data_train, data_test

class ScaffoldKFold(_BaseKFold):

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None, smiles_list=[]):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        self.smiles_list = smiles_list
        self.get_scaffold_lists()
    
    def get_scaffold_lists(self):
        scaffolds = {}
        for idx,smiles in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [idx]
            else:
                scaffolds[scaffold].append(idx)

        self.scaffold_lists = list(scaffolds.values())

    def _make_test_folds(self, X, y=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        n_splits = self.n_splits
        test_size = n_samples // n_splits + 1
        rng = check_random_state(self.random_state)
        y = np.asarray(y)

        y = column_or_1d(y)
        
    
        if self.shuffle:
            rng.shuffle(self.scaffold_lists)
        _scaffold_lists = self.scaffold_lists

        test_folds = np.full(n_samples, -1, dtype='i')
        for fold in range(self.n_splits):
            test_idx = []
            taken = []
            for idx, scaff in enumerate(_scaffold_lists):
                if len(test_idx) + len(scaff) <= test_size:
                    test_idx.extend(scaff)
                    taken.append(idx)
            test_folds[test_idx] = fold
            _scaffold_lists = [i for j, i in enumerate(_scaffold_lists) if j not in set(taken)]
            # print(len(_scaffold_lists))

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)

class RepeatedScaffoldKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None, smiles_list=[]):
        super().__init__(
            ScaffoldKFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits, smiles_list=smiles_list
        )

class StratifiedScaffoldKFold(GroupsConsumerMixin, _BaseKFold):

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None, scaff_based: Literal['median', 'mean'] = 'median'):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        # self.smiles_list = smiles_list
        self.scaff_based = scaff_based
        if scaff_based not in ['median','mean']:
            raise ValueError('scaff_based is expected to be "median" or "mean". The assigned value was {val}'.format(val=repr(scaff_based)))
        # # self.scaff_n_splits = scaff_n_splits
        # # self.get_scaffold_lists()
    
    # def get_scaffold_lists(self):
    #     scaffolds = {}
    #     for idx,smiles in enumerate(self.smiles_list):
    #         mol = Chem.MolFromSmiles(smiles)
    #         scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    #         if scaffold not in scaffolds:
    #             scaffolds[scaffold] = [idx]
    #         else:
    #             scaffolds[scaffold].append(idx)

    #     self.scaffold_lists = list(scaffolds.values())

    def _iter_test_indices(self, X, y, groups):
        # Implementation is based on this kaggle kernel:
        # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        # and is a subject to Apache 2.0 License. You may obtain a copy of the
        # License at http://www.apache.org/licenses/LICENSE-2.0
        # Changelist:
        # - Refactored function to a class following scikit-learn KFold
        #   interface.
        # - Added heuristic for assigning group to the least populated fold in
        #   cases when all other criteria are equal
        # - Swtch from using python ``Counter`` to ``np.unique`` to get class
        #   distribution
        # - Added scikit-learn checks for input: checking that target is binary
        #   or multiclass, checking passed random state, checking that number
        #   of splits is less than number of members in each class, checking
        #   that least populated class has more members than there are splits.
        rng = check_random_state(self.random_state)
        y = np.asarray(y)

        y = column_or_1d(y)
        
        scaffolds = defaultdict(list)
        for idx, scaff_idx in enumerate(groups):
            scaffolds[scaff_idx].append(idx)
        scaffold_lists = list(scaffolds.values())
        
        n_bins = int(np.floor(len(scaffold_lists)/np.array([len(i) for i in scaffold_lists],dtype='i').mean()))
        discretizer=KBinsDiscretizer(n_bins=n_bins,encode='ordinal',strategy='quantile')
        
        scaff_act=[]
        for scaff in scaffold_lists:
            scaff_act.append(y[scaff])
        
        if self.scaff_based == 'median':
            scaff_act_val =[np.median(i) for i in scaff_act]
        elif self.scaff_based == 'mean':
            scaff_act_val = [np.mean(i) for i in scaff_act]
        
        scaff_gr=discretizer.fit_transform(np.array(scaff_act_val).reshape(-1,1))[:,0]
        
        assert len(scaff_gr) == len(scaffold_lists)
        
        bin_assign = np.full(len(X),-1,dtype='i')
        for i, bin in enumerate(scaff_gr):
            bin_assign[scaffold_lists[i]] = scaff_gr[i]
        
        assert -1 not in bin_assign
        
        y = bin_assign
        
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        n_smallest_class = np.min(y_cnt)
        if self.n_splits > n_smallest_class:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (n_smallest_class, self.n_splits),
                UserWarning,
            )
        n_classes = len(y_cnt)

        _, groups_inv, groups_cnt = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)

        if self.shuffle:
            rng.shuffle(y_counts_per_group)

        # Stable sort to keep shuffled order for groups with the same
        # class distribution variance
        sorted_groups_idx = np.argsort(
            -np.std(y_counts_per_group, axis=1), kind="mergesort"
        )

        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(
                y_counts_per_fold=y_counts_per_fold,
                y_cnt=y_cnt,
                group_y_counts=group_y_counts,
            )
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

        for i in range(self.n_splits):
            test_indices = [
                idx
                for idx, group_idx in enumerate(groups_inv)
                if group_idx in groups_per_fold[i]
            ]
            yield test_indices

    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = (
                fold_eval < min_eval
                or np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold

class RepeatedStratifiedScaffoldKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None, scaff_based: Literal['median', 'mean'] = 'median'):
        super().__init__(
            StratifiedScaffoldKFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits, scaff_based=scaff_based
        )

def get_scaff(smiles_list):
    scaffolds = {}
    for idx,smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)
    return scaffolds

def get_scaffold_groups(smiles_list):
    scaffolds = get_scaff(smiles_list)

    scaffold_lists = list(scaffolds.values())
    groups = np.full(len(smiles_list),-1,dtype='i')
    for i, scaff in enumerate(scaffold_lists):
        groups[scaff] = i
    
    assert -1 not in groups
    return groups

def StratifiedScaffold_train_test_split(data: pd.DataFrame, smiles_col: str, activity_col: str, n_splits: int = 8,
                                        scaff_based: Literal['median', 'mean'] = 'median',
                                        random_state=None, 
                                        shuffle=True):
    cv = StratifiedScaffoldKFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state, scaff_based=scaff_based)
    groups = get_scaffold_groups(data[smiles_col].to_list())
    y = data[activity_col].to_numpy(dtype=float)
    X = data.drop([activity_col,smiles_col], axis=1).to_numpy()
    train_idx, test_idx = next(cv.split(X,y,groups))
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    
    return train, test