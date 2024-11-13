#!/usr/bin/env python3
"""A wrapper script for calling data processing functions.
All modules MUST have an add_arguments function
which adds the subcommand to the subparser.
"""

import argparse
import sys

from mwrpy import process_mwrpy, utils


def main(args):
    args = _parse_args(args)
    process_mwrpy.main(args)


def _parse_args(args):
    parser = argparse.ArgumentParser(description="MWRpy processing main wrapper.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["process", "plot", "no-plot", "reprocess"],
        default="process",
        help="Command to execute.",
    )
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
        "--products",
        help="Products to be processed, e.g., 1C01, 2I02, multi.\
                            Default is all collocated products.",
        type=lambda s: s.split(","),
        default=[
            "1C01",
            "single",
            "multi",
        ],
    )
    group.add_argument(
        "--start",
        type=str,
        metavar="YYYY-MM-DD",
        help="Starting date. Default is current day - 1 (included).",
        default=utils.get_date_from_past(1),
    )
    group.add_argument(
        "--stop",
        type=str,
        metavar="YYYY-MM-DD",
        help="Stopping date. Default is current day + 1 (excluded).",
        default=utils.get_date_from_past(-1),
    )
    group.add_argument(
        "-d",
        "--date",
        type=str,
        metavar="YYYY-MM-DD",
        help="Single date to be processed.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
