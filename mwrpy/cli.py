#!/usr/bin/env python3
"""A wrapper script for calling data processing functions.
All modules MUST have an add_arguments function
which adds the subcommand to the subparser.
"""

import argparse
import sys

from mwrpy import process_mwrpy  # , utils

# import warnings


# warnings.simplefilter("ignore", UserWarning)
# warnings.simplefilter("ignore", RuntimeWarning)

modules = {
    "process": process_mwrpy,
}


def main(args):
    args = _parse_args(args)
    cmd = args.cmd
    modules[cmd].main(args)


def _parse_args(args):
    parser = argparse.ArgumentParser(
        description="MWRpy processing main wrapper.", epilog="Have fun!"
    )
    subparsers = parser.add_subparsers(
        title="Command", help="Command to execute.", required=True, dest="cmd"
    )
    for module in modules.values():
        subparsers = module.add_arguments(subparsers)
    group = parser.add_argument_group(title="General options")
    group.add_argument(
        "-s",
        "--site",
        required=True,
        help="Site to process data from, e.g. juelich",
        type=str,
    )
    group.add_argument(
        "-p",
        "--product",
        required=True,
        help="Product to be processed, e.g., 1C01, 2I02, 2P03, stats.",
    )
    # group.add_argument(
    #    "--start",
    #    type=str,
    #    metavar="YYYY-MM-DD",
    #    help="Starting date. Default is current day - 1 (included).",
    #    default=utils.get_date_from_past(1),
    # )
    # group.add_argument(
    #    "--stop",
    #    type=str,
    #    metavar="YYYY-MM-DD",
    #    help="Stopping date. Default is current day + 1 (excluded).",
    #    default=utils.get_date_from_past(-1),
    # )
    # group.add_argument(
    #    "-d",
    #    "--date",
    #    type=str,
    #    metavar="YYYY-MM-DD",
    #    help="Single date to be processed.",
    # )
    return parser.parse_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
