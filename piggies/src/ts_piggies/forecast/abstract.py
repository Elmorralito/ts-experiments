from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


class AbstractModelWrapperConfig(BaseModel):

    learning_rate: float = Field(default=0.03, ge=0.001, le=0.1)
    batch_size: int = Field(default=64, ge=16, le=128)
    max_epochs: int = Field(default=150, ge=50, le=1200)


class ForecastResult(BaseModel):
    """Result from a single forecast run"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str
    config: Dict[str, Any]
    forecast: List[float | np.float64]
    dates: List[str | int | pd.Timestamp]  # Store as list of strings
    train_loss: float | None = None

    @field_validator("forecast", mode="before")
    @classmethod
    def convert_forecast(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @field_validator("dates", mode="before")
    @classmethod
    def convert_dates(cls, v):
        if isinstance(v, np.ndarray):
            # Convert datetime64 to strings
            return [str(pd.to_datetime(d)) for d in v]
        return v


class ProbabilisticForecastResult(BaseModel):
    """Probabilistic forecast result from ensemble of models"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    dates: List[str | int | pd.Timestamp]
    mean: List[float | np.float64]
    median: List[float | np.float64]
    std: List[float | np.float64]
    quantiles: Dict[str, List[float | np.float64]]
    individual_forecasts: List[ForecastResult]
    n_models: int

    @field_validator("dates", mode="before")
    @classmethod
    def convert_dates(cls, v):
        if isinstance(v, np.ndarray):
            # Convert datetime64 to strings
            return [str(pd.to_datetime(d)) for d in v]
        return v


class ForecastModelWrapper(ABC):
    __slots__ = (
        "data",
        "config",
        "model",
        "trainer",
        "train_loss",
        "max_encoder_length",
        "max_prediction_length",
        "n_simulations",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        config: AbstractModelWrapperConfig,
        max_encoder_length: int,
        max_prediction_length: int,
        n_simulations: int,
    ):
        if not isinstance(config, AbstractModelWrapperConfig):
            raise ValueError("config must be a TFTConfig")

        if not isinstance(data, pd.DataFrame) or getattr(data, "empty", True):
            raise ValueError("data must be a pandas DataFrame")

        if not isinstance(max_encoder_length, int) or max_encoder_length <= 0:
            raise ValueError("max_encoder_length must be a positive integer")

        if not isinstance(max_prediction_length, int) or max_prediction_length <= 0:
            raise ValueError("max_prediction_length must be a positive integer")

        if not isinstance(n_simulations, int) or n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer")

        self.data = data.copy()
        self.config = config
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.n_simulations = n_simulations
        self.training_dataset: TimeSeriesDataSet | None = None
        self.model: TemporalFusionTransformer | None = None
        self.trainer: Trainer | None = None
        self.train_loss: float | None = None

    @abstractmethod
    def fit(self) -> None:
        """Train the model"""

    @abstractmethod
    def predict(self, *quantiles: Sequence[float]) -> ForecastResult:
        """Predict the model"""

    @abstractmethod
    def predict_simulations(
        self, *quantiles: Sequence[float]
    ) -> ProbabilisticForecastResult:
        """Predict the model using simulations"""

    @abstractmethod
    def fit_predict(
        self, *quantiles: Sequence[float], simulations: bool = False
    ) -> ForecastResult | ProbabilisticForecastResult:
        """Fit the model and predict the model"""
