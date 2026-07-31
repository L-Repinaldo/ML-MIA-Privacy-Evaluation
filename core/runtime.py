import warnings


def configure_runtime():

    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns",
        category=UserWarning,
        module="sklearn.preprocessing._encoders",
    )

    warnings.filterwarnings(
        "ignore",
        message="`sklearn.utils.parallel.delayed` should be used",
    )