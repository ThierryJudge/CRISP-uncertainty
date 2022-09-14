import argparse
from typing import Type, Union

import yaml


class StoreDictKeyPair(argparse.Action):
    """Action that can parse a python dictionary from comma-separated key-value pairs passed to the parser."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parses comma-separated key-value pairs passed to the parser into a dict, adding it to the namespace.

        Args:
            parser: Parser object being parsed.
            namespace: Namespace produced from parsing the arguments.
            values: Values specified for the option using the current action.
            option_string: Option flag.
        """
        # Hack converting `values` to a YAML document to use the YAML parser's type inference
        yaml_str = values.replace("=", ": ").replace(",", "\n")
        args_map = yaml.safe_load(yaml_str)

        setattr(namespace, self.dest, args_map)


def int_or_float(val: str) -> Union[int, float]:
    """Parses a string as either an integer or a float.

    Args:
        val: String representation of an integer/float to parse.

    Returns:
        Integer/float value parsed from the string.
    """
    try:
        cast_val = int(val)
    except ValueError:
        try:
            cast_val = float(val)
        except ValueError:
            raise
    return cast_val


def get_classpath_group(parser: argparse.ArgumentParser, cls: Type) -> argparse._ArgumentGroup:
    """Fetches an argument group named after the provided class' qualified name from the argument parser.

    Hack because it relies on some private class types and fields of the argument parser.

    Args:
        parser: Argument parser from which to get/create an argument group.
        cls: Class whose qualified name to use for the argument group.

    Returns:
        An argument group in the parser named after the provided class' qualified name.
    """
    cls_path = f"{cls.__module__}.{cls.__qualname__}"
    cls_group = [arg_group for arg_group in parser._action_groups if arg_group.title == cls_path]
    if cls_group:  # If an existing group matches the requested class
        cls_group = cls_group[0]
    else:  # Else create a new group for the requested class
        cls_group = parser.add_argument_group(cls_path)
    return cls_group
