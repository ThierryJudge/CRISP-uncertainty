import logging
from pathlib import Path
from typing import Union


def configure_logging(
    logger_level: Union[int, str] = logging.NOTSET,
    log_to_console: bool = True,
    console_level: Union[int, str] = logging.WARNING,
    log_file: Path = None,
    file_level: Union[int, str] = logging.INFO,
    formatter: logging.Formatter = None,
) -> None:
    """Configures a standardized way of logging for the library.

    Args:
        logger_level: Default level of events propagated by loggers.
        log_to_console: Whether the loggers should display the messages to the console.
        console_level: Minimal level of events to log to the console.
        log_file: Path to the file loggers should write to, if any.
        file_level: Minimal level of events to log to the file.
        formatter: If provided, overrides the default formatter set for all the handlers.
    """
    handlers = []

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        handlers.append(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that the log directory exists
        file_handler = logging.FileHandler(str(log_file), mode="w")
        file_handler.setLevel(file_level)
        handlers.append(file_handler)

    fmt_kwargs = {}
    if formatter:
        for handler in handlers:
            handler.setFormatter(formatter)
    else:
        fmt_kwargs.update(
            {"format": "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"}
        )

    logging.basicConfig(**fmt_kwargs, handlers=handlers, force=True, level=logger_level)
