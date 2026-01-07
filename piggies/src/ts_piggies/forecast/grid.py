import itertools
import logging
from enum import Enum
from itertools import product
from typing import Any, Dict, Iterator, List, Literal, Self, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from .abstract import (  # ProbabilisticForecastResult,
    AbstractModelWrapperConfig,
    ForecastModelWrapper,
    ForecastResult,
)

logger = logging.getLogger(__name__)


class GridSearchConfig(BaseModel):

    error_metric: Literal["MAE", "MAPE", "MASE"] = "MAE"
    validation_length: float = Field(default=0.2, ge=0.0, le=1.0)
    config_type: Type[AbstractModelWrapperConfig]
    model_type: Type[ForecastModelWrapper]
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    hyperparameter_grid: Dict[str, List[Any]]
    max_encoder_length: int
    max_prediction_length: int
    n_simulations: int = 1000


class ErrorMetric(Enum):
    MAE = "MAE"
    MAPE = "MAPE"
    MSE = "MSE"

    def calculate_mae(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        multioutput: Literal["uniform_average", "raw_values"] = "uniform_average",
    ) -> np.float64:
        return np.mean(mean_absolute_error(y_true, y_pred, multioutput=multioutput))

    def calculate_mape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        multioutput: Literal["uniform_average", "raw_values"] = "uniform_average",
    ) -> np.float64:
        return np.mean(
            mean_absolute_percentage_error(y_true, y_pred, multioutput=multioutput)
        )

    def calculate_mse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        multioutput: Literal["uniform_average", "raw_values"] = "uniform_average",
    ) -> np.float64:
        return np.mean(mean_squared_error(y_true, y_pred, multioutput=multioutput))

    def calculate_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        multioutput: Literal["uniform_average", "raw_values"] = "uniform_average",
    ) -> np.float64:
        metric = getattr(self, f"calculate_{self.value.lower()}")
        return metric(y_true, y_pred, multioutput=multioutput)


class GridSearch:

    __slots__ = (
        "data",
        "config",
        "best_model_config",
        "best_model_error",
        "error_multioutput",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        config: GridSearchConfig,
        error_multioutput: Literal["uniform_average", "raw_values"] = "uniform_average",
    ):
        self.data = data
        self.config = config
        self.best_model_config: AbstractModelWrapperConfig | None = None
        self.best_model_error: np.float64 = np.inf
        self.error_multioutput = error_multioutput

    def fit_predict(self) -> ForecastResult:
        if self.best_model_config is None:
            raise ValueError("No best model found")

        model = self.config.model_type(
            data=self.data,
            config=self.best_model_config,
            max_encoder_length=self.config.max_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            n_simulations=self.config.n_simulations,
        )
        logger.info("Fitting and predicting model with config:\n%s", model.config)
        return model.fit_predict(*self.config.quantiles)

    def search(
        self, max_models: int | None = None, random_sample: bool = False
    ) -> Self:
        prediction_length = len(self.validation_data.index)
        logger.info("Starting grid search with %s models", max_models or "all")
        for idx, config in enumerate(itertools.islice(self.grid, max_models)):
            logger.info("Validating iteration %s of %s", idx + 1, max_models or "all")
            model = self.config.model_type(
                data=self.training_data,
                config=config,
                max_encoder_length=self.config.max_encoder_length,
                max_prediction_length=prediction_length,
                n_simulations=self.config.n_simulations,
            )
            error = self.validate(model)
            logger.debug("Error: %s", error)
            if error < self.best_model_error:
                logger.info("New best error: %s for config: %s", error, config)
                self.best_model_error = error
                self.best_model_config = config

        logger.info("Grid search completed")
        logger.info("Best error: %.6f", self.best_model_error)
        logger.info("Best model config:\n%s", self.best_model_config)
        return self

    def validate(self, model: ForecastModelWrapper) -> np.float64:
        logger.debug("Validating model config: %s", model.config)
        testing_data = model.fit_predict(*self.config.quantiles)
        logger.debug(
            "Calculating error, using error metric: %s", self.config.error_metric
        )
        error = self.config.error_metric.calculate_error(
            self.validation_data.values,
            testing_data.values,
            multioutput=self.error_multioutput,
        )
        return error

    @property
    def training_data(self) -> pd.DataFrame:
        return self.data.iloc[: -int(self.config.validation_length * len(self.data))]

    @property
    def validation_data(self) -> pd.DataFrame:
        return self.data.iloc[-int(self.config.validation_length * len(self.data)) :]

    @property
    def grid(self) -> Iterator[AbstractModelWrapperConfig]:
        keys = list(self.config.hyperparameter_grid.keys())
        values = [self.config.hyperparameter_grid[key] for key in keys]
        for combination in product(*values):
            config_dict = dict(zip(keys, combination))
            yield self.config.config_type.model_validate(config_dict)
