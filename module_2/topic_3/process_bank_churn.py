'''
Utilities for preprocessing the Bank Churn dataset.

- preprocess_data(raw_df): fit preprocessing on train split and return processed
  train/val matrices + fitted transformers.
- preprocess_new_data(new_df, input_cols, scaler, encoder): apply already-fitted
  transformers to new/unseen data (e.g., test.csv).

For the experiments in this homework we drop identifiers and the 'Surname' column
to keep the model easier to interpret
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

DEFAULT_TARGET_COL = 'Exited'
DEFAULT_DROP_COLS = ['id', 'CustomerId', 'Surname']


def _safe_onehot_encoder() -> OneHotEncoder:
    '''
    Create OneHotEncoder with backward-compatible args
    '''
    return OneHotEncoder(handle_unknown='ignore', sparse_output=False)


def get_input_cols(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET_COL,
        drop_cols: Optional[List[str]] = None,
) -> List[str]:
    '''
    Build the list of feature columns used for training

    Parameters
    ----------
    df:
        Full raw dataframe that contains features + target.
    target_col:
        Name of the target column (default: 'Exited').
    drop_cols:
        Columns to exclude from features (identifiers, etc.).

    Returns
    -------
    input_cols:
        List of feature column names to be used as X (before any encoding/scaling).
    '''
    drop_cols = DEFAULT_DROP_COLS if drop_cols is None else drop_cols
    return [c for c in df.columns if c != target_col and c not in drop_cols]


def split_train_val(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET_COL,
        test_size: float = 0.2,
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Split the raw dataframe into train and validation parts with stratification.

    Returns X_train_raw, X_val_raw, y_train, y_val.
    '''
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val


def build_numeric_pipeline(scale_numeric: bool = True) -> Pipeline:
    '''
    Build preprocessing pipeline for numeric features.

    If scale_numeric is False, only imputes missing values (median).
    '''
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps=steps)


def build_categorical_pipeline() -> Pipeline:
    '''
    Build preprocessing pipeline for categorical features:
    - impute missing values with most frequent category
    - one-hot encode (unknown categories ignored)
    '''
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', _safe_onehot_encoder()),
        ]
    )


def _infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    '''
    Infer numeric and categorical columns from dtypes in the provided dataframe.
    '''
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # keep bool as categorical (often 0/1 stored as bool); if you want it numeric, cast before calling
    return numeric_cols, categorical_cols


def fit_transformers(
        X_train_inputs: pd.DataFrame,
        scale_numeric: bool = True,
) -> Tuple[Pipeline, Pipeline, List[str], List[str]]:
    '''
    Fit numeric and categorical pipelines on training data.

    Returns
    -------
    scaler:
        Fitted numeric pipeline (imputer + optional StandardScaler).
    encoder:
        Fitted categorical pipeline (imputer + OneHotEncoder).
    numeric_cols, categorical_cols:
        Column names used for each pipeline.
    '''
    numeric_cols, categorical_cols = _infer_feature_types(X_train_inputs)

    scaler = build_numeric_pipeline(scale_numeric=scale_numeric)
    encoder = build_categorical_pipeline()

    if numeric_cols:
        scaler.fit(X_train_inputs[numeric_cols])
    if categorical_cols:
        encoder.fit(X_train_inputs[categorical_cols])

    # Attach the column lists for reliable inference-time transforms
    setattr(scaler, 'numeric_cols_', numeric_cols)
    setattr(encoder, 'categorical_cols_', categorical_cols)

    # Store output feature names (handy for debugging / feature importance later)
    onehot: OneHotEncoder = encoder.named_steps['onehot']
    cat_feature_names = (
        onehot.get_feature_names_out(categorical_cols).tolist() if categorical_cols else []
    )
    out_feature_names = numeric_cols + cat_feature_names
    setattr(encoder, 'output_feature_names_', out_feature_names)

    return scaler, encoder, numeric_cols, categorical_cols


def transform_inputs(
        X_inputs: pd.DataFrame,
        input_cols: List[str],
        scaler: Pipeline,
        encoder: Pipeline,
) -> np.ndarray:
    '''
    Transform raw inputs into a single numeric feature matrix.

    Parameters
    ----------
    X_inputs:
        Raw inputs dataframe (may include extra columns; we select input_cols).
    input_cols:
        List of columns to use as features.
    scaler:
        Fitted numeric pipeline.
    encoder:
        Fitted categorical pipeline.

    Returns
    -------
    X_processed:
        Numpy array of shape (n_samples, n_features_after_encoding)
    '''
    X = X_inputs[input_cols].copy()

    numeric_cols: List[str] = getattr(scaler, 'numeric_cols_', [])
    categorical_cols: List[str] = getattr(encoder, 'categorical_cols_', [])

    parts = []

    if numeric_cols:
        X_num = scaler.transform(X[numeric_cols])
        # StandardScaler returns np.ndarray; SimpleImputer too
        parts.append(np.asarray(X_num, dtype=float))

    if categorical_cols:
        X_cat = encoder.transform(X[categorical_cols])
        parts.append(np.asarray(X_cat, dtype=float))

    if not parts:
        # Should not happen for this dataset, but keep it safe.
        return np.empty((len(X), 0), dtype=float)

    return np.hstack(parts)


def preprocess_data(raw_df: pd.DataFrame,
                    target_col: str = DEFAULT_TARGET_COL,
                    drop_cols: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    scaler_numeric: bool = False
                    ) -> tuple[ndarray, Series, ndarray, Series, list[str], Pipeline, Pipeline]:
    '''
    Full preprocessing for training:
    1) drop columns (for feature selection only)
    2) split into train/val (stratified)
    3) fit preprocessors on train
    4) transform train and val into numeric matrices

    Parameters
    ----------
    raw_df:
        Raw dataframe containing features + target column.
    target_col:
        Target column name (default: 'Exited').
    drop_cols:
        Columns to exclude from X (default: ['id','CustomerId','Surname']).
    test_size:
        Share of validation split.
    random_state:
        Random seed for reproducible split.
    scaler_numeric:
        If True, apply StandardScaler to numeric columns.
        For DecisionTree models scaling is not required, so default is False.

    Returns
    -------
    X_train:
        Processed training feature matrix.
    train_targets:
        y_train.
    X_val:
        Processed validation feature matrix.
    val_targets:
        y_val.
    input_cols:
        List of columns used as X before encoding/scaling.
    scaler:
        Fitted numeric preprocessing pipeline.
    encoder:
        Fitted categorical preprocessing pipeline.
    '''
    drop_cols = DEFAULT_DROP_COLS if drop_cols is None else drop_cols

    # Split first (to avoid leakage), then pick columns consistently
    X_train_raw, X_val_raw, y_train, y_val = split_train_val(
        raw_df, target_col=target_col, test_size=test_size, random_state=random_state
    )

    # Determine input columns from the *full* raw df (matches notebook logic)
    input_cols = get_input_cols(raw_df, target_col=target_col, drop_cols=drop_cols)

    train_inputs = X_train_raw[input_cols]
    val_inputs = X_val_raw[input_cols]

    scaler, encoder, _, _ = fit_transformers(train_inputs, scale_numeric=scaler_numeric)

    X_train = transform_inputs(train_inputs, input_cols, scaler, encoder)
    X_val = transform_inputs(val_inputs, input_cols, scaler, encoder)

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str], scaler: Pipeline, encoder: Pipeline) -> np.ndarray:
    '''
    Preprocess new/unseen data using already-fitted transformers.

    Use this for test.csv or any future inference.

    Parameters
    ----------
    new_df:
        New raw dataframe with (at least) columns from input_cols.
    input_cols:
        Feature columns used during training.
    scaler:
        Fitted numeric preprocessing pipeline (from preprocess_data).
    encoder:
        Fitted categorical preprocessing pipeline (from preprocess_data).

    Returns
    -------
    X_new:
        Processed feature matrix compatible with the model trained on X_train.
    '''
    return transform_inputs(new_df, input_cols=input_cols, scaler=scaler, encoder=encoder)
