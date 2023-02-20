import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from importlib import import_module
from typing import Dict, Union, Callable, Optional, Tuple, Any, List
from numpy.typing import ArrayLike
from functools import partial
from ._utils import _index_X, _index_y, _index_groups, hasparam, suppress_all_warnings
from ._types import Estimator, CrossValidator
from datetime import datetime
from pathlib import Path
import json


with open(Path(__file__).parent / "config.json", "r") as file:
    CONFIG = json.load(file)


def compile_scores(
    config: Dict[str, Union[str, dict]],
    y_true: ArrayLike,
    estimations: Dict[str, ArrayLike],
) -> Dict[str, Union[Callable, dict]]:
    compiled_scores = {}
    for category in config.keys():
        compiled_scores[category] = {}
        for key, item in config[category].items():
            base_func = getattr(import_module(item["source"]), item["method"])
            params = {**item["params"]}
            params.update({"y_true": y_true})

            if hasparam(base_func, "y_score") and "y_score" in estimations.keys():
                params.update({"y_score": estimations["y_score"]})

            if hasparam(base_func, "y_pred") and "y_pred" in estimations.keys():
                params.update({"y_pred": estimations["y_pred"]})

            compiled_func = partial(base_func, **params)
            compiled_scores[category].update({key: compiled_func})
    return compiled_scores


def execute_scores(
    tree: Dict[str, Union[dict, Callable]]
) -> Dict[str, Union[dict, ArrayLike]]:
    exhausted_tree = {}
    for key, item in tree.items():
        if isinstance(item, dict):
            exhausted_tree.update({key: execute_scores(item)})
        else:
            try:
                exhausted_tree.update({key: item()})
            except Exception as e:
                print(f"{key}:{str(e)}")
    return exhausted_tree


def train_step(
    model,
    X: ArrayLike,
    y: ArrayLike,
    groups: Optional[ArrayLike],
    tracing_func: Optional[Callable] = None,
    **fit_kwargs,
) -> Tuple[Any, Any]:
    if hasparam(model.fit, "groups"):
        fit_kwargs.update({"groups": groups})

    trained_model = model.fit(X, y, **fit_kwargs)
    trace = tracing_func(trained_model) if tracing_func else None
    return trained_model, trace


def inject_leaves(tree: dict, leaf: dict) -> dict:
    injected_tree = {}
    for key, item in tree.items():
        if isinstance(item, dict):
            injected_tree.update({key: inject_leaves(item, leaf)})
        else:
            injected_tree.update({key: item, **leaf})
    return injected_tree


def test_step(
    trained_model, X: ArrayLike, methods: Dict[str, str]
) -> Dict[str, ArrayLike]:
    inference = {}
    for method, score in methods.items():
        if hasattr(trained_model, method):
            inference[score] = getattr(trained_model, method)(X)
    return inference


def leaf_to_frame(
    leaf: dict, ignore: Optional[List[str]] = None
) -> Union[pd.DataFrame, pd.Series, dict]:
    """Transforms one-level-deep branches of a dictionary into series or dataframes.

    Particularly, if a branch has items like:
    1. Arraylike: a dataframe is returned where each key becomes a column
    2. floats: a series where each key becomes the value of the index
    3. dict: repeats the above untils a one-level-deep branch is reached.


    Parameters
    ----------
    leaf : dict
    ignore : Optional[List[str]], optional
        Ignores particular keys, by default None

    Returns
    -------
    Union[pd.DataFrame, pd.Series, dict]
        Pandas converted branches dictionary.
    """

    def is_leaf_instance(tree, instance, ignore=None):
        ignore = [] if ignore is None else ignore
        instances = [
            isinstance(tree[key], instance) for key in tree.keys() if key not in ignore
        ]
        return all(instances)

    if is_leaf_instance(leaf, np.ndarray, ignore):
        leaf_df = pd.DataFrame(leaf)
    elif is_leaf_instance(leaf, float, ignore):
        leaf_df = pd.Series(leaf)
    elif is_leaf_instance(leaf, dict, ignore):
        leaf_df = {key: leaf_to_frame(item, ignore) for key, item in leaf.items()}
    else:
        raise ValueError("'leaf' can only be one of [dict,np.ndarray,float]")
    return leaf_df


def leaf_to_frame_from_list(
    list_of_leaves: List[dict], ignore: Optional[List[str]] = None
) -> List[dict]:
    """Maps 'leaf_to_frame' to each element of list 'list_of_leaves'.

    Parameters
    ----------
    list_of_leaves : List[dict]
    ignore : Optional[List[str]], optional
        Ignores particular keys, by default None

    Returns
    -------
    list[dict]
    """
    return list(map(partial(leaf_to_frame, ignore=ignore), list_of_leaves))


def merge_nested_results(
    leaf_1: Union[pd.DataFrame, pd.Series],
    leaf_2: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    """Concatenates 'leaf_1' and 'leaf_2'.

    Lazy wrapper that applies 'pd.concat' depending on whether the inputs as 'pd.Series' or 'pd.DataFrame'.

    Parameters
    ----------
    leaf_1 : Union[pd.DataFrame,pd.Series]
    leaf_2 : Union[pd.DataFrame,pd.Series]

    Returns
    -------
    Union[pd.DataFrame,pd.Series]
    """
    if isinstance(leaf_1, pd.DataFrame) and isinstance(leaf_2, pd.DataFrame):
        merged_leaf = pd.concat([leaf_1, leaf_2], axis=0).reset_index(drop=True)
    elif isinstance(leaf_1, pd.Series) and isinstance(leaf_2, pd.Series):
        merged_leaf = pd.concat([leaf_1, leaf_2], axis=1).T
    elif isinstance(leaf_1, pd.DataFrame) and isinstance(leaf_2, pd.Series):
        merged_leaf = pd.concat([leaf_1, leaf_2.to_frame().T], axis=0).reset_index(
            drop=True
        )
    return merged_leaf


def zip2dicts(tree_1: dict, tree_2: dict, merge_func: Callable = lambda *x: x) -> dict:
    """Given two dictionaries with equal keys and equal types of items, returns a dictionary with same keys as the input and with concatenated items.

    Parameters
    ----------
    tree_1 : dict
    tree_2 : dict
    merge_func : Callable, optional
        Function defining how to merge the leaves of tree_1 and tree_2, by default lambda*x:x

    Returns
    -------
    dict
        Merged tree.
    """
    merged_tree = {}
    for (key1, item1), (_, item2) in zip(tree_1.items(), tree_2.items()):
        if isinstance(item1, dict) and isinstance(item2, dict):
            # if branch, nest operation untill leaf is reached
            merged_tree[key1] = zip2dicts(item1, item2, merge_func=merge_func)
        else:
            # if leaves, merge them
            merged_tree[key1] = merge_func(item1, item2)
    return merged_tree


def zip_dicts(*trees: List[dict], merge_func: Callable = lambda *x: x) -> dict:
    """Extension of 'zip2dicts' to an iterable of dictionaries.

    Current implementation may be taxing on memory for large inputs.

    Parameters
    ----------
    merge_func : Callable, optional
        Function defining how to merge the leaves of tree_1 and tree_2, by default lambda*x:x

    Returns
    -------
    dict
        Merged dictionary.
    """
    merged_tree = trees[0]
    for current_tree in trees[1:]:
        merged_tree = zip2dicts(merged_tree, current_tree, merge_func=merge_func)
    return merged_tree


def explode_tree_of_scores_into_dataframes(
    scores: List[Dict[str, Union[dict, float, ArrayLike]]],
) -> Dict[str, pd.DataFrame]:
    """Converts jsonable dictionaties into pandas dataframe.

    Given a list of dictionaries 'scores', where each leaf is either an Arraylike or float,
    returns a single dictionary where all leaves are combained into dataframes.
    This roughly equates to a transformation:
    '[{x:1,y:1},{x:2,y2}]->{x:[1,2],y:[1,2]}'


    Parameters
    ----------
    scores : List[Dict[str, Union[dict, float, ArrayLike]]]


    Returns
    -------
    Dict[str, pd.DataFrame]
    """

    def post_process_indexes(
        tree: Dict[str, Union[dict, pd.DataFrame]], index: List[str]
    ) -> Dict[str, Union[dict, pd.DataFrame]]:
        processed_tree = {}
        for key, item in tree.items():
            if isinstance(item, dict):
                processed_tree[key] = post_process_indexes(item, index)
            else:
                processed_tree[key] = item.set_index(index)
        return processed_tree

    leaves = leaf_to_frame_from_list(scores, ignore=["name", "fold", "side"])
    results = zip_dicts(*leaves, merge_func=merge_nested_results)
    return post_process_indexes(results, index=["name", "fold", "side"])


def name_func(name) -> str:
    return name if name else f"model_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"


def time_func():
    return datetime.datetime.now()


@suppress_all_warnings
def crossvalidation(
    config: dict,
    mappings: dict,
    model: Estimator,
    X: ArrayLike,
    y: ArrayLike,
    groups: Optional[ArrayLike] = None,
    cv: CrossValidator = KFold(),
    tracing_func: Optional[Callable] = None,
    name: Optional[str] = None,
    progress_bar: bool = True,
    **fit_kwargs,
) -> Tuple[Dict[str, Union[dict, pd.DataFrame]], dict, dict]:
    """Base crossvalidation function.

    Splits `X` and `y`, and `groups` if given, in different train/test folds as defined by `cv`.
    After each training, `model` is fed to `tracing_func` and its results are stored in `fold_traces`.
    At the same time, scores defined in `config` are estimated both from the training and test fold.



    Parameters
    ----------
    config : dict
        A dicitonary containing the scores to compute.
    mappings : dict
        Sklearn nomenclature mappings, for example the `predict` method returns a `y_pred` variable.
    model : Estimator
        Sklearn-like estimator.
    X : ArrayLike
        Input data.
    y : ArrayLike
        Targets.
    groups : Optional[ArrayLike], optional
        Group associated to each sample, by default None
    cv : CrossValidator, optional
        Sklearn-like cross validator, by default KFold()
    tracing_func : Optional[Callable], optional
        A callable which takes as input a trained estmator. There are no constraints in the form of its output(s), by default None
    name : Optional[str], optional
        Name of the experiment, by default None
    progress_bar : bool, optional
        If True, shows the progress of each fold, by default True

    Returns
    -------
    Tuple[Dict[str, Union[dict, pd.DataFrame]],dict,dict]
        The output of crossvalidate is split in three dicitonaries:
        #. fold_scores: A dictionary where the keys of its leaves represent the name of the scoring metric and the item a dataframe with the results.
        #. fold_traces: A dicitonary where each key is the fold number and each item the result of the `tracing_func`, if provided.
        #. fold_indexes: A dicitonary where each key is the fold number and the items are the indexesof each train/test split with thre predicitons and predict probas.

    """
    fold_scores = []
    fold_traces = {}
    fold_indexes = {}

    name = name_func(name)
    n_splits = cv.get_n_splits(X, y, groups)
    pbar = tqdm(
        enumerate(cv.split(X, y, groups=groups)),
        total=n_splits,
        disable=not (progress_bar),
        desc=name,
    )

    for fold, (train_ix, test_ix) in pbar:
        # slice data according to validations folds
        X_train, y_train, groups_train = (
            _index_X(X, train_ix),
            _index_y(y, train_ix),
            _index_groups(groups, train_ix),
        )
        X_test, y_test, groups_test = (
            _index_X(X, test_ix),
            _index_y(y, test_ix),
            _index_groups(groups, test_ix),
        )

        # model training
        trained_model, trace = train_step(
            model,
            X_train,
            y_train,
            groups_train,
            tracing_func=tracing_func,
            **fit_kwargs,
        )

        # scoring train set
        train_inferences = test_step(trained_model, X_train, mappings)
        compiled_train_scores = compile_scores(config, y_train, train_inferences)
        train_scores = execute_scores(compiled_train_scores)
        train_scores = inject_leaves(
            train_scores,
            {"fold": fold, "side": "train", "name": name},
        )  # ensures dictionary is 'columns' oriented

        # scoring test set
        test_inferences = test_step(trained_model, X_test, mappings)
        compiled_test_scores = compile_scores(config, y_test, test_inferences)
        test_scores = execute_scores(compiled_test_scores)
        test_scores = inject_leaves(
            test_scores,
            {"fold": fold, "side": "test", "name": name},
        )  # ensures dictionary is 'columns' oriented

        # storing traces and scores
        fold_scores += [train_scores, test_scores]
        fold_traces[fold] = trace
        fold_indexes[fold] = {
            "train": {"ix": train_ix, **train_inferences},
            "test": {"ix": test_ix, **test_inferences},
        }

    return (
        explode_tree_of_scores_into_dataframes(fold_scores),
        fold_traces,
        fold_indexes,
    )


def crossvalidate_classification(*args, **kwargs):
    return crossvalidation(
        CONFIG["CLASSIFICATION"], CONFIG["MAPPINGS"], *args, **kwargs
    )


def crossvalidate_regression(*args, **kwargs):
    return crossvalidation(CONFIG["REGRESSION"], CONFIG["MAPPINGS"], *args, **kwargs)
