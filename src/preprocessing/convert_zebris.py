""" This script converts Zebris XML files into gzip compressed json files.
    Since Zebris produces nested data structures, the raw data cannot be stored in simple CSV files.
    The XML files turned out to be very large and need a lot of time to be loaded/parsed.
    Loading JSON files is significantly faster and can be performed via build-in functions in Pandas,
    without the need to care about parsing.
    However, JSON files are still quite big due to repetitive keywords. This can cause long loading
    times when reading from disk/network. Thus, the files are gzip compressed. This leads to fast
    loading and small files.
"""

import os
import fnmatch
import xmltodict
from collections import OrderedDict
import json
import gzip

if __name__ == "__main__":
    directory = "./data/raw/TRIPOD"

    for root, d_names, f_names in os.walk(directory):
        for f in f_names:
            file_name = os.path.join(root, f)
            if fnmatch.fnmatch(f, "zebris_raw.xml"):
                print(file_name)
                parsed_dict = xmltodict.parse(open(file_name, "rb"))

                clean_zebris = {}
                clean_zebris["cell_count"] = OrderedDict(
                    [
                        ("x", int(parsed_dict["measurement"]["cell_count"]["x"])),
                        ("y", int(parsed_dict["measurement"]["cell_count"]["y"])),
                    ]
                )
                clean_zebris["cell_size"] = OrderedDict(
                    [
                        (
                            "x",
                            float(
                                parsed_dict["measurement"]["movements"]["movement"][
                                    "clips"
                                ]["clip"][0]["cell_size"]["x"]
                            ),
                        ),
                        (
                            "y",
                            float(
                                parsed_dict["measurement"]["movements"]["movement"][
                                    "clips"
                                ]["clip"][0]["cell_size"]["y"]
                            ),
                        ),
                    ]
                )
                clean_zebris["frequency"] = int(
                    parsed_dict["measurement"]["movements"]["movement"]["clips"][
                        "clip"
                    ][0]["frequency"]
                )
                clean_zebris["pressure_unit"] = parsed_dict["measurement"]["movements"][
                    "movement"
                ]["clips"]["clip"][0]["units"]
                clean_zebris["time_unit"] = "s"
                clean_zebris["velocity_unit"] = parsed_dict["measurement"]["movements"][
                    "movement"
                ]["clips"]["clip"][1]["units"]
                clean_zebris["sample_count"] = int(
                    parsed_dict["measurement"]["movements"]["movement"]["clips"][
                        "clip"
                    ][0]["count"]
                )
                clean_zebris["begin"] = float(
                    parsed_dict["measurement"]["movements"]["movement"]["clips"][
                        "clip"
                    ][0]["begin"]
                )
                clean_zebris["samples"] = []

                for pressure, velocity in zip(
                    parsed_dict["measurement"]["movements"]["movement"]["clips"][
                        "clip"
                    ][0]["data"]["quant"],
                    parsed_dict["measurement"]["movements"]["movement"]["clips"][
                        "clip"
                    ][1]["data"]["quant"],
                ):
                    if pressure["cells"]:
                        clean_zebris["samples"].append(
                            OrderedDict(
                                [
                                    (
                                        "origin",
                                        OrderedDict(
                                            [
                                                ("x", int(pressure["cell_begin"]["x"])),
                                                ("y", int(pressure["cell_begin"]["y"])),
                                            ]
                                        ),
                                    ),
                                    (
                                        "size",
                                        OrderedDict(
                                            [
                                                ("x", int(pressure["cell_count"]["x"])),
                                                ("y", int(pressure["cell_count"]["y"])),
                                            ]
                                        ),
                                    ),
                                    (
                                        "pressure",
                                        [float(v) for v in pressure["cells"].split()],
                                    ),
                                    ("velocity", float(velocity)),
                                ]
                            )
                        )
                    else:
                        clean_zebris["samples"].append(
                            OrderedDict(
                                [
                                    ("origin", OrderedDict([("x", None), ("y", None)])),
                                    ("size", OrderedDict([("x", 0), ("y", 0)])),
                                    ("pressure", []),
                                    ("velocity", float(velocity)),
                                ]
                            )
                        )

                with gzip.open(
                    os.path.join(root, "zebris_raw.json.gz"), "wt", encoding="ascii"
                ) as outfile:
                    json.dump(clean_zebris, outfile, indent=4)

            elif fnmatch.fnmatch(f, "zebris_steps.xml"):
                print(file_name)
                parsed_dict = xmltodict.parse(open(file_name, "rb"))
                clean_zebris = {}
                clean_zebris["cell_count"] = OrderedDict(
                    [
                        ("x", int(parsed_dict["measurement"]["cell_count"]["x"])),
                        ("y", int(parsed_dict["measurement"]["cell_count"]["y"])),
                    ]
                )
                clean_zebris["cell_size"] = OrderedDict(
                    [
                        (
                            "x",
                            float(
                                parsed_dict["measurement"]["movements"]["movement"][
                                    "clips"
                                ]["clip"]["data"]["event"][0]["cell_size"]["x"]
                            ),
                        ),
                        (
                            "y",
                            float(
                                parsed_dict["measurement"]["movements"]["movement"][
                                    "clips"
                                ]["clip"]["data"]["event"][0]["cell_size"]["y"]
                            ),
                        ),
                    ]
                )
                clean_zebris["frequency"] = int(
                    parsed_dict["measurement"]["movements"]["movement"]["clips"][
                        "clip"
                    ]["data"]["event"][0]["rollover"]["frequency"]
                )
                clean_zebris["unit"] = parsed_dict["measurement"]["movements"][
                    "movement"
                ]["clips"]["clip"]["data"]["event"][0]["rollover"]["units"]
                clean_zebris["event_count"] = int(
                    parsed_dict["measurement"]["movements"]["movement"]["clips"][
                        "clip"
                    ]["count"]
                )
                clean_zebris["events"] = []

                for event in parsed_dict["measurement"]["movements"]["movement"][
                    "clips"
                ]["clip"]["data"]["event"]:
                    begin = float(event["begin"])
                    end = float(event["end"])
                    side = event["side"]
                    heel = OrderedDict(
                        [
                            ("x", float(event["heel"]["x"])),
                            ("y", float(event["heel"]["y"])),
                        ]
                    )
                    toe = OrderedDict(
                        [
                            ("x", float(event["toe"]["x"])),
                            ("y", float(event["toe"]["y"])),
                        ]
                    )
                    maximum = OrderedDict(
                        [
                            (
                                "origin",
                                OrderedDict(
                                    [
                                        ("x", int(event["max"]["cell_begin"]["x"])),
                                        ("y", int(event["max"]["cell_begin"]["y"])),
                                    ]
                                ),
                            ),
                            (
                                "size",
                                OrderedDict(
                                    [
                                        ("x", int(event["max"]["cell_count"]["x"])),
                                        ("y", int(event["max"]["cell_count"]["y"])),
                                    ]
                                ),
                            ),
                            (
                                "pressure",
                                [float(v) for v in event["max"]["cells"].split()],
                            ),
                        ]
                    )
                    rollover = {}
                    rollover["sample_count"] = int(event["rollover"]["count"])
                    rollover["size"] = OrderedDict(
                        [
                            ("x", int(event["rollover"]["cell_count"]["x"])),
                            ("y", int(event["rollover"]["cell_count"]["y"])),
                        ]
                    )
                    rollover["samples"] = []
                    for data in event["rollover"]["data"]["quant"]:
                        rollover["samples"].append(
                            OrderedDict(
                                [
                                    (
                                        "origin",
                                        OrderedDict(
                                            [
                                                ("x", int(data["cell_begin"]["x"])),
                                                ("y", int(data["cell_begin"]["y"])),
                                            ]
                                        ),
                                    ),
                                    (
                                        "size",
                                        OrderedDict(
                                            [
                                                ("x", int(data["cell_count"]["x"])),
                                                ("y", int(data["cell_count"]["y"])),
                                            ]
                                        ),
                                    ),
                                    (
                                        "pressure",
                                        [float(v) for v in data["cells"].split()],
                                    ),
                                ]
                            )
                        )
                    clean_zebris["events"].append(
                        {
                            "begin": begin,
                            "end": end,
                            "side": side,
                            "heel": heel,
                            "toe": toe,
                            "maximum": maximum,
                            "rollover": rollover,
                        }
                    )

                with gzip.open(
                    os.path.join(root, "zebris_steps.json.gz"), "wt", encoding="ascii"
                ) as outfile:
                    json.dump(clean_zebris, outfile, indent=4)
