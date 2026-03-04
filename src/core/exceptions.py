class AppBaseException(Exception):
    """Base exception for application domain errors."""


class InvalidTaxiTypeError(AppBaseException, ValueError):
    """Raised when an unsupported taxi fleet type is specified."""


class DataExtractionError(AppBaseException, RuntimeError):
    """Raised when remote TLC parquet dataset extraction fails."""


class InsufficientDataError(AppBaseException, ValueError):
    """Raised when a dataset contains fewer samples than required for modeling."""


class ModelNotLoadedError(AppBaseException, RuntimeError):
    """Raised when an inference request is dispatched to an uninitialized model."""


class InvalidSplitError(AppBaseException, ValueError):
    """Raised when a temporal split leaves one of the resulting partitions empty."""


class CorruptedMetadataError(AppBaseException, ValueError):
    """Raised when a persisted model metadata file cannot be parsed into a known shape."""


class CorruptedPromotionLogError(AppBaseException, ValueError):
    """Raised when the promotion history cannot be parsed."""


class IncompatibleModelError(AppBaseException, RuntimeError):
    """Raised when a stored artifact's feature contract diverges from the running code."""


class VersionAlreadyExistsError(AppBaseException, RuntimeError):
    """Raised when saving would overwrite an artifact already stored under the same version."""


class GeocodingError(AppBaseException, RuntimeError):
    """Raised when the external geocoding service cannot be reached or understood."""
