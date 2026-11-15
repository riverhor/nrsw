"""Normal-referenced spectral weighting for registered document layouts."""
from .api import Calibration, calibrate, evaluate, fit, load_model, predict, save_model
from .registry import LayoutRegistry, UnknownLayoutError
from .schema import FeatureDataset, load_feature_dataset, save_feature_dataset

__version__ = "0.4.0"
__all__ = ["Calibration", "FeatureDataset", "LayoutRegistry", "UnknownLayoutError", "calibrate", "evaluate", "fit", "load_feature_dataset",
           "load_model", "predict", "save_feature_dataset", "save_model"]
