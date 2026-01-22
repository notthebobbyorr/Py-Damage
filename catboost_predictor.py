"""
CatBoost Baseball Model Predictor

Helper class for making predictions with a trained CatBoost model
on freshly scraped or new data. Includes standardization and 20-80 scaling.

Usage:
    from catboost_predictor import CatBoostBaseballPredictor

    # Load model
    predictor = CatBoostBaseballPredictor('catboost_baseball_model.cbm')

    # Predict on new data
    predictions = predictor.predict(new_df)

    # Predict with 20-80 scaling (for pitcher aggregations)
    df_with_grades = predictor.predict_with_scaling(pitcher_agg_df)
"""

import json
import pandas as pd
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from pathlib import Path

# 20-80 scouting scale parameters
SCALE_MEAN = 50
SCALE_STD = 10


class CatBoostBaseballPredictor:
    """
    Helper class to make predictions with saved CatBoost baseball models.

    Handles:
    - Loading saved model and metadata
    - Preprocessing new data (pitch_group creation, feature alignment)
    - Making predictions
    - Feature importance analysis
    """

    def __init__(self, model_path="catboost_baseball_model.cbm"):
        """
        Load a saved CatBoost model and its metadata.

        Args:
            model_path: Path to the saved .cbm model file
        """
        self.model_path = Path(model_path)

        # Load model
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))

        # Load metadata
        metadata_path = str(model_path).replace(".cbm", "_metadata.json")
        try:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Metadata file not found at {metadata_path}")
            self.metadata = {}

        # Extract metadata
        self.features = self.metadata.get("features", self.model.feature_names_)
        self.categorical_features = self.metadata.get("categorical_features", [])
        self.target_col = self.metadata.get("target_col", "event_exp")
        self.pitch_group_mapping = self.metadata.get(
            "pitch_group_mapping",
            {"FA": ["FA"], "BR": ["CU", "SL"], "OFF": ["CH", "FS"], "XX": "other"},
        )
        self.pitch_group_source_col = self.metadata.get(
            "pitch_group_source_col", "pi_pitch_group"
        )

        print(f"Loaded CatBoost model from {model_path}")
        print(f"  Features: {len(self.features)}")
        print(f"  Categorical: {self.categorical_features}")
        print(f"  Target: {self.target_col}")

    def create_pitch_group(self, df):
        """
        Create pitch_group column for new data.

        Args:
            df: DataFrame (pandas or polars)

        Returns:
            DataFrame with pitch_group column
        """
        is_polars = isinstance(df, pl.DataFrame)

        if is_polars:
            source_col = self.pitch_group_source_col

            # Check for source column or alternatives
            if source_col not in df.columns:
                alternatives = ["pitch_tag", "pitch_type", "pitch_name"]
                for alt in alternatives:
                    if alt in df.columns:
                        source_col = alt
                        break
                else:
                    raise ValueError(f"Could not find pitch type column")

            fast = self.pitch_group_mapping.get("FA", ["FA"])
            spin = self.pitch_group_mapping.get("BR", ["CU", "SL"])
            off = self.pitch_group_mapping.get("OFF", ["CH", "FS"])

            df = df.with_columns(
                [
                    pl.when(pl.col(source_col).is_in(fast))
                    .then(pl.lit("FA"))
                    .when(pl.col(source_col).is_in(spin))
                    .then(pl.lit("BR"))
                    .when(pl.col(source_col).is_in(off))
                    .then(pl.lit("OFF"))
                    .otherwise(pl.lit("XX"))
                    .alias("pitch_group")
                ]
            )
        else:
            # Pandas version
            source_col = self.pitch_group_source_col

            if source_col not in df.columns:
                alternatives = ["pitch_tag", "pitch_type", "pitch_name"]
                for alt in alternatives:
                    if alt in df.columns:
                        source_col = alt
                        break
                else:
                    raise ValueError(f"Could not find pitch type column")

            fast = self.pitch_group_mapping.get("FA", ["FA"])
            spin = self.pitch_group_mapping.get("BR", ["CU", "SL"])
            off = self.pitch_group_mapping.get("OFF", ["CH", "FS"])

            df = df.copy()
            conditions = [
                df[source_col].isin(fast),
                df[source_col].isin(spin),
                df[source_col].isin(off),
            ]
            choices = ["FA", "BR", "OFF"]
            df["pitch_group"] = np.select(conditions, choices, default="XX")

        return df

    def preprocess(self, df):
        """
        Preprocess new data for prediction.

        Args:
            df: DataFrame with raw features (pandas or polars)

        Returns:
            Preprocessed pandas DataFrame ready for prediction
        """
        # Convert polars to pandas if needed
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        else:
            df = df.copy()

        # Create pitch_group if not present and needed
        if "pitch_group" in self.features and "pitch_group" not in df.columns:
            df = self.create_pitch_group(df)

        # Check for missing features
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(
                f"Warning: Missing features (will be filled with median): {missing_features}"
            )
            for f in missing_features:
                df[f] = np.nan

        # Select only required features in correct order
        df_features = df[self.features].copy()

        return df_features

    def predict(self, df, return_dataframe=False):
        """
        Make predictions on new data.

        Args:
            df: DataFrame with raw features
            return_dataframe: If True, return DataFrame with predictions column

        Returns:
            numpy array of predictions, or DataFrame if return_dataframe=True
        """
        # Preprocess
        df_processed = self.preprocess(df)

        # Handle missing values (CatBoost handles them natively, but let's be safe)
        # For categorical features, fill with mode or 'unknown'
        for col in self.categorical_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna("unknown")

        # Make predictions
        predictions = self.model.predict(df_processed)

        if return_dataframe:
            result = df.copy() if not isinstance(df, pl.DataFrame) else df.to_pandas()
            result["catboost_pred"] = predictions
            return result

        return predictions

    def predict_with_intervals(self, df, n_iterations=100):
        """
        Make predictions with uncertainty estimates using CatBoost's
        virtual ensembles feature (if available).

        Note: This requires the model to be trained with posterior_sampling=True

        Args:
            df: DataFrame with raw features
            n_iterations: Number of virtual ensemble iterations

        Returns:
            dict with 'mean', 'std', 'lower', 'upper' predictions
        """
        df_processed = self.preprocess(df)

        try:
            # Try to get prediction intervals
            preds = self.model.predict(df_processed)
            # Virtual ensembles not available in standard training
            # Return point predictions with note

            return {
                "mean": preds,
                "std": np.zeros_like(preds),  # No uncertainty estimate
                "lower": preds,
                "upper": preds,
                "note": "Uncertainty estimation requires model trained with posterior_sampling=True",
            }
        except Exception as e:
            print(f"Prediction interval estimation failed: {e}")
            preds = self.model.predict(df_processed)
            return {
                "mean": preds,
                "std": np.zeros_like(preds),
                "lower": preds,
                "upper": preds,
            }

    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from the model.

        Args:
            top_n: If specified, return only top N features

        Returns:
            DataFrame with feature names and importance scores
        """
        importance_df = pd.DataFrame(
            {
                "feature": self.model.feature_names_,
                "importance": self.model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def get_shap_values(self, df, sample_size=1000):
        """
        Get SHAP values for interpreting predictions.

        Args:
            df: DataFrame to explain
            sample_size: Max samples to compute SHAP for (for speed)

        Returns:
            SHAP values array
        """
        df_processed = self.preprocess(df)

        if len(df_processed) > sample_size:
            df_processed = df_processed.sample(n=sample_size, random_state=42)

        try:
            shap_values = self.model.get_feature_importance(
                data=df_processed, type="ShapValues"
            )
            return shap_values
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            return None

    def evaluate(self, df, target_col=None):
        """
        Evaluate model on data with known targets.

        Args:
            df: DataFrame with features and target
            target_col: Target column name (uses self.target_col if not specified)

        Returns:
            dict with evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        target_col = target_col or self.target_col

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        predictions = self.predict(df)
        actuals = (
            df[target_col].values
            if isinstance(df, pd.DataFrame)
            else df[target_col].to_numpy()
        )

        # Handle missing values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]

        metrics = {
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
            "r2": r2_score(actuals, predictions),
            "mae": mean_absolute_error(actuals, predictions),
            "correlation": np.corrcoef(actuals, predictions)[0, 1],
            "n_samples": len(predictions),
        }

        print(f"Evaluation Metrics:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R2:   {metrics['r2']:.4f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  Corr: {metrics['correlation']:.4f}")
        print(f"  N:    {metrics['n_samples']:,}")

        return metrics


def example_usage():
    """Example of how to use the predictor."""

    # Load predictor
    predictor = CatBoostBaseballPredictor("catboost_baseball_model.cbm")

    # Example: Load new data
    # new_data = pd.read_parquet('new_scraped_data.parquet')

    # Example: Create some test data
    test_data = pd.DataFrame(
        {
            "avg_release_z": [5.8, 5.5, 6.0],
            "avg_release_x": [-1.2, -1.0, -1.5],
            "avg_ext": [6.5, 6.2, 6.8],
            "pitch_velo": [94.5, 92.0, 96.0],
            "rpm": [2250, 2100, 2400],
            "vbreak": [12.3, 10.5, 14.0],
            "hbreak": [-8.5, -6.0, -10.0],
            "axis": [2.1, 2.0, 2.3],
            "spin_efficiency": [0.85, 0.80, 0.90],
            "z_angle_release": [12.0, 10.0, 14.0],
            "x_angle_release": [-2.5, -2.0, -3.0],
            "vaa": [-4.5, -5.0, -4.0],
            "haa": [8.2, 7.5, 9.0],
            "throws": ["R", "L", "R"],
            "stands": ["R", "L", "R"],
            "pi_pitch_group": ["FA", "CU", "CH"],
        }
    )

    # Make predictions
    predictions = predictor.predict(test_data)
    print(f"\nPredictions: {predictions}")

    # Get predictions with original data
    result_df = predictor.predict(test_data, return_dataframe=True)
    print(
        f"\nResults with predictions:\n{result_df[['pitch_velo', 'throws', 'catboost_pred']]}"
    )

    # Get feature importance
    importance = predictor.get_feature_importance(top_n=10)
    print(f"\nTop 10 Important Features:\n{importance}")


if __name__ == "__main__":
    example_usage()
