import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold

from . import _utils as ut
from . import _types as tp
from ._controllers import validate_input

from itertools import product
from warnings import warn

from typing import Optional, Union, Type, List, Any
from numpy.typing import ArrayLike, DTypeLike

Covariates = Type[Union[ArrayLike, pd.DataFrame, List[List[float]]]]
Targets = Type[Union[ArrayLike, pd.Series, List[float]]]
Groups = Type[Union[ArrayLike, DTypeLike, pd.Series, List[Any]]]


class WalkForwardCV:
    """[summary]"""

    def __init__(
        self,
        min_train_size: int = 1,
        max_train_size: Optional[int] = None,
        gap: int = 0,
        test_size: int = 1,
        step: int = 1,
    ):
        assert min_train_size > 0
        if max_train_size:
            assert max_train_size > 0
            assert max_train_size >= min_train_size

        assert test_size > 0
        assert gap >= 0
        assert step > 0

        self.max_train_size = max_train_size
        self.min_train_size = min_train_size
        self.gap = gap
        self.step = step
        self.test_size = test_size

    def _process_groups(self, groups: Groups):

        _groups = ut.make_array_from_listlike(groups)  # making groups a numpy array
        sorted_groups = np.arange(
            _groups.min(), _groups.max(), 1
        )  # sorting unique elements
        sequence_length = sorted_groups.shape[0]  # sequence length
        return _groups, sorted_groups, sequence_length

    def _get_past_and_new_data(self, ix: int, old_ix: int, max_size: int, step: int):
        """
        Given two timestamps, ix and old_ix, and a buffer size max_size, the
        method identifies what are the common values between the previous iteration
        and the current.
        This method is the core of WalkForwardCV and aims at speeding up
        computation of train and test indexes.


        Parameters
        ----------
        ix : int
            Current timestamp.
        old_ix : int
            Previous timestamp.
        max_size : int
            maximum size of the desired buffer.
        step : int
            Difference between consecutive ix, please note the the difference
            between old_ix and ix may be greater or shorter than step.

        Returns
        -------
        new_data : TYPE
            DESCRIPTION.
        past_data : TYPE
            DESCRIPTION.
        current_size : TYPE
            DESCRIPTION.

        """
        # current size of the array to generate
        # if there is no max_size, the current size is from the index 0 to the current index
        # if the current index is lower than the max size, the current size is from the index 0 to the current index
        # if the current index is greater than max size, the current size if from the current index minus max size to the current index
        current_size = ix if (max_size is None) or (ix < max_size) else max_size

        if max_size:
            # if max size is not empy, the current size is capped to max_size
            if step < max_size:
                new_data = np.arange(old_ix, ix, 1)  # new data in the buffer
                past_data = np.arange(
                    old_ix + step - current_size, old_ix, 1
                )  # data tha can be retrieved from the old buffer
            else:
                # if step is greater than max size, than there is no common values between steps in the current buffer
                new_data = np.arange(ix - current_size, ix, 1)
                past_data = None
        else:
            # if current size is not bounded by max_size, all past data is always recycled
            new_data = np.arange(old_ix, ix, 1)
            past_data = np.arange(0, old_ix, 1) if old_ix > 0 else None

        return new_data, past_data, current_size

    def _update_indexer(self, groups, sorted_groups, new_data, past_data, past_dict):
        if (len(past_dict) == 0) and (past_data is not None):
            new_dict = ut.findall(
                groups, sorted_groups[np.concatenate((new_data, past_data))]
            )  # when past_dict is empty we need to retrieve past data as well.
        else:
            new_dict = ut.findall(
                groups, sorted_groups[new_data]
            )  # find index of new data in groups

        new_ix = [
            i for (_, i) in new_dict.items()
        ]  # retrieve indexes of the new samples

        past_ix = [
            past_dict[k]
            for k in sorted_groups[past_data]
            if (past_data is not None) and (len(past_dict) > 0)
        ]
        past_dict.update(new_dict)
        return np.concatenate((*new_ix, *past_ix))

    def get_n_splits(
        self,
        X: Optional[Covariates] = None,
        y: Optional[Targets] = None,
        groups: Optional[Groups] = None,
    ):
        validate_input(X, y, groups, ignore_X=True, ignore_y=True, ignore_groups=False)

        _groups, sorted_groups, sequence_length = self._process_groups(groups)
        return len(
            range(
                self.min_train_size,
                sequence_length - self.test_size - self.gap,
                self.step,
            )
        )

    def split(
        self,
        X: Covariates,
        y: Optional[Targets] = None,
        groups: Optional[Groups] = None,
    ):
        validate_input(X, y, groups, ignore_X=True, ignore_y=True, ignore_groups=False)

        _groups, sorted_groups, sequence_length = self._process_groups(groups)

        told = 0
        past_train_dict = {}
        past_test_dict = {}
        test_overhead = self.test_size + self.gap
        for t in range(
            self.min_train_size, sequence_length - self.test_size - self.gap, self.step
        ):
            new_train_data, past_train_data, train_size = self._get_past_and_new_data(
                t, told, self.max_train_size, self.step
            )
            new_test_data, past_test_data, test_size = self._get_past_and_new_data(
                t + test_overhead,
                t + test_overhead - self.step,
                self.test_size,
                self.step,
            )

            told = t

            train_ix = self._update_indexer(
                _groups, sorted_groups, new_train_data, past_train_data, past_train_dict
            )
            test_ix = self._update_indexer(
                _groups, sorted_groups, new_test_data, past_test_data, past_test_dict
            )

            yield train_ix, test_ix


# FIX
class StackedCV:
    def __init__(
        self,
        cv_list: List[tp.CrossValidator] = [
            StratifiedKFold(n_splits=5),
            GroupKFold(n_splits=5),
        ],
    ):

        self.cv_list = cv_list

    def get_n_splits(self, X=None, y=None, groups=None):
        validate_input(X, y, groups, ignore_X=True, ignore_y=True, ignore_groups=True)
        slices = self._prepare_groups(groups)
        return np.prod(
            [
                ccv.get_n_splits(X=X, y=y, groups=clice)
                for ccv, clice in zip(self.cv_list, slices)
            ]
        )

    def split(self, X=None, y=None, groups=None):
        validate_input(X, y, groups, ignore_X=True, ignore_y=True, ignore_groups=True)
        slices = self._prepare_groups(groups)

        splitters = [
            ccv.split(X, y, groups=cslices)
            for (ccv, cslices) in zip(self.cv_list, slices)
        ]
        for macro_ix in product(*splitters):
            # this is wrong, missing consequentiality
            tr_curr, ts_curr = macro_ix[0][0], macro_ix[0][1]
            train_ix = [
                tr_curr := np.intersect1d(tr_curr, val[0]) for val in macro_ix[1:]
            ][0]
            test_ix = [
                ts_curr := np.intersect1d(ts_curr, val[1]) for val in macro_ix[1:]
            ][0]
            yield train_ix, test_ix

    def _prepare_groups(self, groups):
        if groups is None:
            slices = [None for _ in self.cv_list]
        else:
            slices = ut.transpose_list(groups)
            assert len(slices) == len(self.cv_list)
        return slices


class TimeSeriesKFold:
    """
    TO BE UPDATED: NOT FUNCTIONAL DUE TO UPDATE TO findall

    CV splitter supporting split for groups of time series.

    Parameters
    ----------
    n_splits : int, optional
        Number of superset splits.
        The default is 10.
    min_train_size : int, optional
        Minimum number of splits to build the train set.
        The default is 1.
    gap : int, optional
        Number of splits separating the train and test set.
        The default is 0.
    max_train_size : Optional[int], optional
        Maximum size of splits to build the test set. NOT IMPLEMENTED YET.
        The default is None.


    ALPHA
    """

    def __init__(
        self,
        n_splits: int = 10,
        min_train_size: int = 1,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ):

        warn("Do not use this old class, trust me.")
        assert isinstance(n_splits, int)
        assert isinstance(min_train_size, int)
        assert isinstance(gap, int)
        # assert isinstance(max_train_size, int) | max_train_size is None

        self.n_splits = n_splits
        self.gap = gap
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size

    def split(self, X=None, y=None, groups: Optional[Union[list, np.ndarray]] = None):
        """


        Parameters
        ----------
        X : TYPE, optional
            Always ignored, exists for compatibility.
        y : TYPE, optional
            Always ignored, exists for compatibility.
        groups : Union[list, np.ndarray], optional
            Array-like of shape {n_samples,} of int or floats, indicating the
            the time-stamp corrisponding to each sample. Time samples can be
            repeated if there are groups.

        Yields
        ------
        train_ix : TYPE
            DESCRIPTION.
        test_ix : TYPE
            DESCRIPTION.

        """
        assert isinstance(groups, np.ndarray) | isinstance(groups, list)

        stamps = np.unique(groups)
        schedule = np.array_split(stamps, self.n_splits)
        for i in range(self.min_train_size, len(schedule) - self.gap):
            train_stamps = np.concatenate(schedule[:i])
            test_stamps = schedule[i + self.gap]

            train_ix = np.array(ut.findall(groups, train_stamps))
            test_ix = np.array(ut.findall(groups, test_stamps))
            yield (train_ix, test_ix)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits - self.min_train_size - self.gap
