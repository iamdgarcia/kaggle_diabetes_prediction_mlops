"""
Data Preprocessing Module

This module provides an MLOps-friendly `DataPreprocessor` that can be
fit on training data, saved as an artifact, and later loaded to perform
inference-time transforms on new (test/inference) datasets. Training and
inference are explicitly separated: training fits the transformer state
from the training data; inference loads that state and only applies
transformations without peeking at test labels or their categories.

Usage (training):
    preproc = DataPreprocessor(config)
    X_train, y_train = preproc.fit_transform(train_df)
    preproc.save("artifacts/preprocessor.joblib")

Usage (inference):
    preproc = DataPreprocessor.load("artifacts/preprocessor.joblib")
    X_test = preproc.transform_for_inference(test_df)

"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """MLOps-capable data preprocessor.

    Key features:
    - `fit` on training dataframe (records category lists, feature names, indices)
    - `transform` to apply same transforms to other frames without leaking
      information from test data
    - `save` / `load` for reuse in inference pipelines
    """

    def __init__(self, config):
        self.config = config
        self.fitted = False
        # learned state
        self.category_map: Dict[str, List] = {}
        self.cat_features_indices: List[int] = []
        self.feature_names: List[str] = []
        self._numerical_to_boolean = getattr(self.config.features, 'NUMERICAL_TO_BOOLEAN', [])
        self._skewed_cols = getattr(self.config.features, 'SKEWED_TO_GAUSS', [])

    # ---------- I/O helpers ----------
    def load_data(self, path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)

    def save(self, path: str) -> None:
        """Save the fitted preprocessor to `path` using joblib."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor to {path}")

    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load a saved preprocessor artifact."""
        logger.info(f"Loading preprocessor artifact from {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a DataPreprocessor instance")
        return obj

    # ---------- Fit / Transform API ----------
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the preprocessor on training dataframe.

        This records:
        - categorical value lists per categorical column
        - feature names and categorical indices
        """
        # Work on a copy
        df = train_df.copy()

        # Convert numerical-to-boolean columns to categorical dtype
        if self._numerical_to_boolean:
            for col in self._numerical_to_boolean:
                if col in df.columns:
                    df[col] = df[col].astype('category')

        # Apply log transform to skewed columns (train only)
        if self._skewed_cols:
            for col in self._skewed_cols:
                if col in df.columns:
                    # Guard against negative values; clip at -1e-6 then log1p
                    df[col] = np.log1p(np.clip(df[col].astype(float), a_min=0.0, a_max=None))

        # Identify object columns and record their categories from training set
        object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in object_cols:
            # Record unique training categories (preserve order)
            cats = pd.Series(df[col].astype('str').fillna('')).unique().tolist()
            self.category_map[col] = cats
            # Apply categorical dtype with learned categories
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

        # Build feature matrix
        if self.config.model.TARGET_COL not in df.columns:
            raise KeyError(f"Target column {self.config.model.TARGET_COL} not found in training data")

        X = df.drop(columns=[self.config.model.TARGET_COL, self.config.model.ID_COL])
        y = df[self.config.model.TARGET_COL]

        self.feature_names = X.columns.tolist()
        self.cat_features_indices = [i for i, c in enumerate(self.feature_names) if X[c].dtype.name == 'category']
        self.fitted = True

        logger.info(f"Fitted preprocessor on data shape: {df.shape}")
        logger.info(f"Found {len(self.category_map)} categorical columns")

        # keep y if needed by user; don't store train data
        return None

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit on `train_df` and return transformed X and y."""
        self.fit(train_df)

        df = train_df.copy()

        # Same operations as in fit
        if self._numerical_to_boolean:
            for col in self._numerical_to_boolean:
                if col in df.columns:
                    df[col] = df[col].astype('category')

        if self._skewed_cols:
            for col in self._skewed_cols:
                if col in df.columns:
                    df[col] = np.log1p(np.clip(df[col].astype(float), a_min=0.0, a_max=None))

        # Apply categorical dtypes using learned categories
        for col, cats in self.category_map.items():
            if col in df.columns:
                df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

        X = df.drop(columns=[self.config.model.TARGET_COL, self.config.model.ID_COL])
        y = df[self.config.model.TARGET_COL]

        # update cat indices in case types changed
        self.feature_names = X.columns.tolist()
        self.cat_features_indices = [i for i, c in enumerate(self.feature_names) if X[c].dtype.name == 'category']

        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations to a dataframe (no label column expected).

        Raises if preprocessor is not fitted.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted or loaded before calling transform()")

        out = df.copy()

        # Convert numerical-to-boolean columns to categorical dtype
        if self._numerical_to_boolean:
            for col in self._numerical_to_boolean:
                if col in out.columns:
                    out[col] = out[col].astype('category')

        # Apply log transform using same columns
        if self._skewed_cols:
            for col in self._skewed_cols:
                if col in out.columns:
                    out[col] = np.log1p(np.clip(out[col].astype(float), a_min=0.0, a_max=None))

        # Apply categorical dtypes using categories learned from training
        for col, cats in self.category_map.items():
            if col in out.columns:
                out[col] = out[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

        # Ensure we return the same feature columns and order the same as training
        missing = [c for c in self.feature_names if c not in out.columns]
        if missing:
            raise KeyError(f"Missing expected feature columns: {missing}")

        out = out[self.feature_names]

        # update cat_features_indices just in case
        self.cat_features_indices = [i for i, c in enumerate(self.feature_names) if out[c].dtype.name == 'category']

        return out

    def transform_for_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Transform incoming test/inference data and drop ID column before returning X_test."""
        out = self.transform(test_df)
        # Drop ID column if present in original test_df
        if self.config.model.ID_COL in out.columns:
            out = out.drop(columns=[self.config.model.ID_COL])
        return out


def create_preprocessor(config):
    return DataPreprocessor(config)

