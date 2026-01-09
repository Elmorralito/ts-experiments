import logging
import random
from enum import Enum
from itertools import islice, product
from typing import Any, Callable, Dict, Iterator, List, Literal, Self, Tuple, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from .abstract import (
    AbstractModelWrapperConfig,
    ForecastModelWrapper,
    ForecastResult,
    ProbabilisticForecastResult,
)

logger = logging.getLogger(__name__)


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


class GridSearchConfig(BaseModel):

    config_type: Type[AbstractModelWrapperConfig]
    model_type: Type[ForecastModelWrapper]
    hyperparameter_grid: Dict[str, List[Any]]
    segment_columns: List[str]
    sort_columns: List[str]
    test_column: str
    max_encoder_length: int
    max_prediction_length: int
    sort_ascending: bool = True
    n_simulations: int = 1000
    error_metric: ErrorMetric = ErrorMetric.MAE
    validation_length: float = Field(default=0.1, ge=0.01, le=0.35)
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]


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
        if not isinstance(config, GridSearchConfig):
            raise ValueError("config must be a GridSearchConfig")

        if not isinstance(data, pd.DataFrame) or getattr(data, "empty", True):
            raise ValueError("data must be a pandas DataFrame")

        self.data = data
        self.config = config
        self.best_model_config: AbstractModelWrapperConfig | None = None
        self.best_model_error: np.float64 = np.inf
        self.error_multioutput = error_multioutput

    def fit_predict(self) -> ForecastResult | ProbabilisticForecastResult:
        if self.best_model_config is None:
            raise ValueError("No best model found")

        logger.info("Training model with full data...")
        model = self.config.model_type(
            data=self.data,
            config=self.best_model_config,
            max_encoder_length=self.config.max_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            n_simulations=self.config.n_simulations,
        )
        logger.info("Fitting and predicting model with config:\n%s", model.config)
        return model.fit_predict(*self.config.quantiles, simulations=True)

    def search(
        self, max_models: int | None = None, random_sample: bool = False
    ) -> Self:
        prediction_length = int(
            self.config.validation_length
            * len(self.data.index)
            / len(self.data[self.config.segment_columns].drop_duplicates().index)
        )
        training_data = self.training_data
        logger.info(
            "Starting grid search with %s models with validation length of %s and training data length of %s",
            max_models or "all",
            prediction_length,
            len(training_data.index),
        )
        logger.info("Prediction length: %s for validation.", prediction_length)

        for idx, config in self.get_grid(random_sample, max_models):
            logger.info("Validating iteration %s of %s", idx + 1, max_models or "all")
            model = self.config.model_type(
                data=training_data,
                config=config,
                max_encoder_length=self.config.max_encoder_length,
                max_prediction_length=prediction_length,
                n_simulations=self.config.n_simulations,
            )
            error = self.validate(model)
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
        testing_data = model.fit_predict(*self.config.quantiles, simulations=False)
        logger.debug(
            "Calculating error, using error metric: %s", self.config.error_metric
        )
        errors = []
        segment_values = self.data.loc[:, self.config.segment_columns].drop_duplicates()
        for _, segment_value in segment_values.iterrows():
            segment_data = (
                self.validation_data.query(
                    " & ".join(
                        [
                            f"({col} == {repr(value)})"
                            for col, value in segment_value.items()
                        ]
                    )
                )
                .loc[:, [self.config.test_column]]
                .values.flatten()
            )
            error = self.config.error_metric.calculate_error(
                segment_data,
                np.array(testing_data.forecast),
                multioutput=self.error_multioutput,
            )
            errors.append(error)

        error = np.min(errors)
        logger.debug("Calculated error: %s", error)
        return error

    def get_dataset(
        self, resize_function: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
    ) -> pd.DataFrame:
        segment_values = self.data.loc[:, self.config.segment_columns].drop_duplicates()
        datasets = []
        for _, segment_value in segment_values.iterrows():
            logger.debug("Getting dataset for segment: %s", segment_value)
            segment_data = self.data.query(
                " & ".join(
                    [
                        f"({col} == {repr(value)})"
                        for col, value in segment_value.items()
                    ]
                )
            ).sort_values(
                by=self.config.sort_columns,
                ignore_index=True,
                ascending=self.config.sort_ascending,
            )
            datasets.append(resize_function(segment_data))

        return pd.concat(datasets, axis=0, ignore_index=True).sort_values(
            by=self.config.sort_columns,
            ignore_index=True,
            ascending=self.config.sort_ascending,
        )

    def get_grid(
        self, random_sample: bool = False, max_models: int | None = None
    ) -> Iterator[Tuple[int, AbstractModelWrapperConfig]]:
        if random_sample:
            random.seed(42)
            shuffled = iter(random.sample(list(self.grid), max_models))
        else:
            shuffled = self.grid

        return enumerate(islice(shuffled, max_models))

    @property
    def training_data(self) -> pd.DataFrame:
        return self.get_dataset(
            lambda df: df.iloc[: -int(self.config.validation_length * len(df.index))]
        )

    @property
    def validation_data(self) -> pd.DataFrame:
        return self.get_dataset(
            lambda df: df.iloc[-int(self.config.validation_length * len(df.index)) :]
        )

    @property
    def grid(self) -> Iterator[AbstractModelWrapperConfig]:
        keys = list(self.config.hyperparameter_grid.keys())
        values = [self.config.hyperparameter_grid[key] for key in keys]
        for combination in product(*values):
            config_dict = dict(zip(keys, combination))
            yield self.config.config_type.model_validate(config_dict)


class SimulationsGrid(GridSearch):
    __slots__ = ("simulations",)

    def __init__(
        self,
        data: pd.DataFrame,
        config: GridSearchConfig,
        error_multioutput: Literal["uniform_average", "raw_values"] = "uniform_average",
    ):
        super().__init__(data, config, error_multioutput)
        self.simulations: List[ProbabilisticForecastResult] | None = None

    def fit_predict(self) -> ProbabilisticForecastResult:
        if not self.simulations:
            raise ValueError("No simulations found")

        dates = self.simulations[0].dates
        n_digits = int(np.log10(len(self.simulations))) + 1
        results = pd.DataFrame(
            data={
                f"fcast_{str(idx + 1).zfill(n_digits)}": model.forecast
                for idx, model in enumerate(self.simulations)
            },
            index=dates,
        )
        return ProbabilisticForecastResult(
            dates=dates,
            mean=results.mean(axis=1).values.tolist(),
            median=results.median(axis=1).values.tolist(),
            std=results.std(axis=1).values.tolist(),
            quantiles={
                str(q): results.quantile(q, axis=1).values.tolist()
                for q in self.config.quantiles
            },
            individual_forecasts=self.simulations,
            n_models=len(self.simulations),
        )

    def search(
        self, max_models: int | None = None, random_sample: bool = False
    ) -> Self:
        self.simulations = []
        for idx, config in self.get_grid(random_sample, max_models):
            logger.info(
                "Running simulation on iteration %s of %s", idx + 1, max_models or "all"
            )
            logger.info("Using config: %s", config)
            model = self.config.model_type(
                data=self.data,
                config=config,
                max_encoder_length=self.config.max_encoder_length,
                max_prediction_length=self.config.max_prediction_length,
                n_simulations=self.config.n_simulations,
            )
            self.simulations.append(
                model.fit_predict(*self.config.quantiles, simulations=False)
            )

        return self
