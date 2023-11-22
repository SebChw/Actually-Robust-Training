"""All predefined exceptions."""


class CheckFailedException(Exception):
    """Exception raised when a check fails."""

    pass


class MissingLogParamsException(Exception):
    """Exception raised when a log is missing parameters."""

    pass


class SavingArtResultsException(Exception):
    """Exception raised when saving ART results fails."""

    pass
