from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from pydantic import Field
from pytorch_forecasting import (
    QuantileLoss,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer

from .abstract import (
    AbstractModelWrapperConfig,
    ForecastModelWrapper,
    ForecastResult,
    ProbabilisticForecastResult,
)


class TFTConfig(AbstractModelWrapperConfig):
    """Temporal Fusion Transformer hyperparameters"""

    hidden_size: int = Field(default=64, ge=16, le=512)
    attention_head_size: int = Field(default=4, ge=1, le=8)
    dropout: float = Field(default=0.15, ge=0.0, le=0.5)
    hidden_continuous_size: int = Field(default=16, ge=8, le=64)
    gradient_clip_val: float = Field(default=0.1, ge=0.0, le=1.0)
    reduce_on_plateau_patience: int = Field(default=6, ge=3, le=10)


class TFTForecastWrapper(ForecastModelWrapper):
    """Wrapper class for Temporal Fusion Transformer forecasting"""

    def __init__(
        self,
        data: pd.DataFrame,
        config: TFTConfig,
        max_encoder_length: int,
        max_prediction_length: int,
        n_simulations: int = 1000,
    ):
        if not isinstance(config, TFTConfig):
            raise ValueError("config must be a TFTConfig")

        super().__init__(
            data, config, max_encoder_length, max_prediction_length, n_simulations
        )

    def fit(self) -> None:
        """Train the TFT model"""
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            self.data,
            time_idx="time_idx",
            target="price",
            group_ids=["series_id"],
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["series_id"],
            time_varying_known_reals=[
                "year",
                "month",
                "production",
                "per_capita_consumption",
            ],
            time_varying_unknown_reals=["price"],
            target_normalizer=GroupNormalizer(groups=["series_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        train_dataloader = self.training_dataset.to_dataloader(
            train=True, batch_size=self.config.batch_size, num_workers=0
        )
        # Create model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=self.config.reduce_on_plateau_patience,
        )

        # Create trainer
        self.trainer = Trainer(
            max_epochs=self.config.max_epochs,
            gradient_clip_val=self.config.gradient_clip_val,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            accelerator="auto",
        )

        # Train model
        self.trainer.fit(self.model, train_dataloaders=train_dataloader)

        # Get training loss
        if (
            hasattr(self.trainer, "callback_metrics")
            and "train_loss" in self.trainer.callback_metrics
        ):
            self.train_loss = float(self.trainer.callback_metrics["train_loss"])

    def _prepare_prediction_data(self):
        """Prepare prediction data for all series"""
        # Sort data by series_id and time_idx to ensure proper ordering
        data_sorted = self.data.sort_values(["series_id", "time_idx"]).copy()
        unique_series = data_sorted["series_id"].unique()

        # Create prediction data for each series
        prediction_data_list = []
        for series_id in unique_series:
            series_data = data_sorted[data_sorted["series_id"] == series_id].copy()
            last_time_idx_series = series_data["time_idx"].max()
            last_eom_series = series_data["eom"].max()

            # Generate future dates starting from the month after the last date
            future_eoms = pd.date_range(
                last_eom_series + pd.offsets.MonthEnd(1),
                periods=self.max_prediction_length,
                freq="ME",
            )

            # Get last known values for time-varying known reals
            last_production = (
                series_data["production"].dropna().iloc[-1]
                if not series_data["production"].dropna().empty
                else (
                    series_data["production"].median()
                    if not series_data["production"].isna().all()
                    else 0
                )
            )
            last_per_capita = (
                series_data["per_capita_consumption"].dropna().iloc[-1]
                if not series_data["per_capita_consumption"].dropna().empty
                else (
                    series_data["per_capita_consumption"].median()
                    if not series_data["per_capita_consumption"].isna().all()
                    else 0
                )
            )
            last_price = (
                series_data["price"].dropna().iloc[-1]
                if not series_data["price"].dropna().empty
                else 0
            )

            # Create prediction rows for future periods
            series_prediction = pd.DataFrame(
                {
                    "eom": future_eoms,
                    "series_id": series_id,
                    "year": future_eoms.year,
                    "month": future_eoms.month,
                    "time_idx": range(
                        last_time_idx_series + 1,
                        last_time_idx_series + 1 + len(future_eoms),
                    ),
                    "production": last_production,
                    "per_capita_consumption": last_per_capita,
                    "price": last_price,  # Will be used as placeholder, model will predict
                }
            )
            prediction_data_list.append(series_prediction)

        prediction_data = pd.concat(prediction_data_list, ignore_index=True)

        # Combine historical and prediction data, ensuring proper sorting
        data_sorted = self.data.sort_values(["series_id", "time_idx"]).copy()
        model_input = pd.concat(
            [
                data_sorted[
                    [
                        "eom",
                        "price",
                        "series_id",
                        "time_idx",
                        "year",
                        "month",
                        "production",
                        "per_capita_consumption",
                    ]
                ],
                prediction_data,
            ],
            ignore_index=True,
        )

        # Sort by series_id and time_idx - CRITICAL for TimeSeriesDataSet
        model_input = model_input.sort_values(["series_id", "time_idx"]).reset_index(
            drop=True
        )

        # Ensure proper data types
        model_input["price"] = model_input["price"].astype(np.float64)
        model_input["production"] = model_input["production"].astype(np.float64)
        model_input["per_capita_consumption"] = model_input[
            "per_capita_consumption"
        ].astype(np.float64)

        return model_input, prediction_data, data_sorted

    def _get_model_predictions(self, model_input, return_quantiles: bool = False):
        """
        Get raw model predictions with all quantiles

        Args:
            model_input: Prepared model input DataFrame
            return_quantiles: If True, tries to get quantile predictions. If False, returns standard predictions.

        Returns:
            predictions_np: numpy array of predictions
            has_quantiles: bool indicating if predictions include quantile dimension
        """
        # Create prediction dataset using the same structure as training
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            model_input,
            predict=True,
            stop_randomization=True,
        )

        # Get predictions - use mode="prediction" to get denormalized predictions
        predict_dataloader = prediction_dataset.to_dataloader(
            train=False, batch_size=1, num_workers=0
        )

        if return_quantiles:
            # Try to get quantile predictions by calling the model directly
            try:
                # Set model to eval mode
                self.model.eval()
                all_predictions = []

                with torch.no_grad():
                    for batch in predict_dataloader:
                        # Get raw output from model
                        x, y = batch
                        output = self.model(x)

                        # Output shape should be [batch_size, prediction_length, num_quantiles]
                        if isinstance(output, torch.Tensor):
                            all_predictions.append(output.detach().cpu().numpy())
                        else:
                            # If output is a dict or tuple, try to extract predictions
                            if isinstance(output, dict):
                                output = output.get(
                                    "prediction", output.get("pred", None)
                                )
                            if output is not None:
                                all_predictions.append(
                                    output.detach().cpu().numpy()
                                    if isinstance(output, torch.Tensor)
                                    else np.array(output)
                                )

                if all_predictions:
                    predictions_np = np.concatenate(all_predictions, axis=0)
                    # Check if we have quantile dimension
                    if predictions_np.ndim == 3 and predictions_np.shape[2] > 1:
                        return predictions_np, True
            except Exception:
                # Fall back to standard predict if direct call fails
                pass

        # Get predictions using standard predict method
        predictions = self.model.predict(predict_dataloader, return_y=False)

        # Extract predictions
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().cpu().numpy()
        elif hasattr(predictions, "numpy"):
            predictions_np = predictions.numpy()
        else:
            predictions_np = np.array(predictions)

        # Check if we have quantile dimension
        has_quantiles = predictions_np.ndim == 3 and predictions_np.shape[-1] > 1

        return predictions_np, has_quantiles

    def _sample_from_quantiles(
        self, quantile_values: np.ndarray, quantile_levels: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Sample from quantile distribution using linear interpolation

        Args:
            quantile_values: Array of quantile values [num_quantiles]
            quantile_levels: Array of quantile levels (e.g., [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
            n_samples: Number of samples to generate

        Returns:
            Array of samples [n_samples]
        """
        # Generate uniform random samples in [0, 1]
        u = np.random.uniform(0, 1, n_samples)

        # Use numpy's interp to interpolate between quantiles
        samples = np.interp(u, quantile_levels, quantile_values)

        return samples

    def _format_dates(self, dates: np.ndarray | pd.Series) -> List[str]:
        """Format dates to ISO format strings"""
        return [
            (
                pd.to_datetime(d).isoformat()
                if isinstance(d, (pd.Timestamp, np.datetime64))
                else str(d)
            )
            for d in dates
        ]

    def _get_median_quantile_index(
        self, predictions_np: np.ndarray, has_quantiles: bool
    ) -> int:
        """Get the index of the median quantile in predictions"""
        if has_quantiles and predictions_np.ndim == 3:
            if hasattr(self.model, "quantiles") and self.model.quantiles is not None:
                try:
                    return list(self.model.quantiles).index(0.5)
                except ValueError:
                    pass
            # Fallback: assume median is the central index
            return predictions_np.shape[-1] // 2
        return 0  # Not used for 2D predictions

    def _extract_predictions(
        self, predictions_np: np.ndarray, median_idx: int
    ) -> np.ndarray:
        """Extract predictions from array, handling different shapes"""
        if predictions_np.ndim == 3:
            return predictions_np[:, :, median_idx]  # [batch_size, pred_length]
        elif predictions_np.ndim == 2:
            return predictions_np  # [batch_size, pred_length]
        elif predictions_np.ndim == 1:
            return predictions_np.reshape(1, -1)  # [1, pred_length]
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions_np.shape}")

    def _get_model_quantiles(self, predictions_np: np.ndarray) -> np.ndarray:
        """Extract quantile levels from model or infer from predictions shape"""
        if hasattr(self.model, "quantiles") and self.model.quantiles is not None:
            return np.array(self.model.quantiles)

        num_quantiles = predictions_np.shape[2]
        if num_quantiles == 7:
            return np.array([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        return np.linspace(0.01, 0.99, num_quantiles)

    def _generate_simulations_from_quantiles(
        self, predictions_np: np.ndarray, model_quantiles: np.ndarray
    ) -> np.ndarray:
        """Generate simulations from quantile predictions using vectorized operations"""
        # predictions_np shape: [batch_size, pred_length, num_quantiles]
        # Average across series (batch dimension) for each time step and quantile
        avg_quantile_preds = np.mean(
            predictions_np, axis=0
        )  # [pred_length, num_quantiles]

        # Generate all samples at once using vectorized operations
        pred_length = avg_quantile_preds.shape[0]
        u = np.random.uniform(
            0, 1, (self.n_simulations, pred_length)
        )  # [n_simulations, pred_length]

        # Interpolate for each time step
        simulations = np.array(
            [
                np.interp(u[:, t], model_quantiles, avg_quantile_preds[t, :])
                for t in range(pred_length)
            ]
        ).T  # [n_simulations, pred_length]

        return simulations

    def _generate_simulations_from_deterministic(
        self, predictions_np: np.ndarray
    ) -> np.ndarray:
        """Generate simulations from deterministic predictions using normal distribution"""
        # Ensure 2D shape: [batch_size, pred_length]
        if predictions_np.ndim != 2:
            predictions_np = predictions_np.reshape(-1, predictions_np.shape[-1])

        # Average across series and compute std for each time step
        mean_preds = np.mean(predictions_np, axis=0)  # [pred_length]
        std_preds = np.std(predictions_np, axis=0)  # [pred_length]
        # Avoid zero std
        std_preds = np.where(std_preds > 0, std_preds, np.abs(mean_preds) * 0.1)

        # Generate samples using vectorized normal distribution
        simulations = np.random.normal(
            mean_preds[None, :],
            std_preds[None, :],
            (self.n_simulations, len(mean_preds)),
        )  # [n_simulations, pred_length]

        return simulations

    def predict_simulations(
        self, quantiles: Optional[List[float]] = None
    ) -> ProbabilisticForecastResult:
        """
        Generate probabilistic forecast with n_simulations samples

        Args:
            quantiles: List of quantiles to compute (default: [0.1, 0.25, 0.5, 0.75, 0.9])

        Returns:
            ProbabilisticForecastResult with mean, std, and quantiles from simulations
        """
        if self.model is None or self.training_dataset is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        # Prepare prediction data
        model_input, prediction_data, data_sorted = self._prepare_prediction_data()
        unique_series = data_sorted["series_id"].unique()

        # Get model predictions
        predictions_np, has_quantiles = self._get_model_predictions(
            model_input, return_quantiles=True
        )

        # Get forecast dates from first series (all should have same dates)
        first_series_pred = prediction_data[
            prediction_data["series_id"] == unique_series[0]
        ].sort_values("eom")
        forecast_dates = pd.to_datetime(first_series_pred["eom"].values)
        formatted_dates = self._format_dates(forecast_dates)

        # Generate simulations based on prediction type
        if has_quantiles and predictions_np.ndim == 3:
            model_quantiles = self._get_model_quantiles(predictions_np)
            simulations_array = self._generate_simulations_from_quantiles(
                predictions_np, model_quantiles
            )
        else:
            simulations_array = self._generate_simulations_from_deterministic(
                predictions_np
            )

        # Compute statistics across simulations (vectorized)
        mean_forecast = np.mean(simulations_array, axis=0).tolist()
        median_forecast = np.median(simulations_array, axis=0).tolist()
        std_forecast = np.std(simulations_array, axis=0).tolist()

        # Compute requested quantiles (vectorized)
        quantile_dict = {
            str(q): np.quantile(simulations_array, q, axis=0).tolist()
            for q in quantiles
        }

        # Create ForecastResult for median (backward compatibility)
        median_forecast_result = ForecastResult(
            model_name="TFT",
            config=self.config.model_dump(),
            forecast=median_forecast,
            dates=formatted_dates,
            train_loss=self.train_loss,
        )

        return ProbabilisticForecastResult(
            dates=formatted_dates,
            mean=mean_forecast,
            median=median_forecast,
            std=std_forecast,
            quantiles=quantile_dict,
            individual_forecasts=[median_forecast_result],
            n_models=1,
        )

    def predict(self, quantiles: Optional[List[float]] = None) -> ForecastResult:
        """
        Generate deterministic forecast predictions (median quantile)

        Args:
            quantiles: Not used in this method (kept for interface compatibility)

        Returns:
            ForecastResult with median predictions
        """
        if self.model is None or self.training_dataset is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        model_input, prediction_data, data_sorted = self._prepare_prediction_data()
        unique_series = data_sorted["series_id"].unique()

        # Get model predictions
        predictions_np, has_quantiles = self._get_model_predictions(
            model_input, return_quantiles=False
        )

        # Get median quantile index and extract predictions
        median_idx = self._get_median_quantile_index(predictions_np, has_quantiles)
        predictions_2d = self._extract_predictions(predictions_np, median_idx)
        # predictions_2d shape: [batch_size, pred_length]

        # Average predictions across all series (vectorized)
        forecast = np.mean(predictions_2d, axis=0)  # [pred_length]

        # Get forecast dates from first series (all series have same dates)
        first_series_pred = prediction_data[
            prediction_data["series_id"] == unique_series[0]
        ].sort_values("eom")
        forecast_dates = pd.to_datetime(first_series_pred["eom"].values)

        return ForecastResult(
            model_name="TFT",
            config=self.config.model_dump(),
            forecast=forecast.tolist(),
            dates=self._format_dates(forecast_dates),
            train_loss=self.train_loss,
        )

    def fit_predict(
        self, *quantiles: Sequence[float], simulations: bool = False
    ) -> ForecastResult | ProbabilisticForecastResult:
        """
        Fit the model and generate predictions in one call

        Args:
            simulations: If True, force ProbabilisticForecastResult. If False, force ForecastResult.
                          If None (default), automatically uses n_simulations: probabilistic if n_simulations > 1
            quantiles: List of quantiles to compute for probabilistic forecast (default: [0.1, 0.25, 0.5, 0.75, 0.9])

        Returns:
            ForecastResult or ProbabilisticForecastResult based on probabilistic parameter and n_simulations
        """
        self.fit()
        return (
            self.predict_simulations(quantiles=quantiles)
            if simulations
            else self.predict(quantiles=quantiles)
        )
