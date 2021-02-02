"""
This module contains functionality that reads data from XML files as they can be exported from the OptoGait software.
This is only used to initially convert the XML files to CSV files, since the Excal XML format is rather cumbersome.
"""

from xml.etree import ElementTree
import pandas as pd


def read_opto_gait_raw_xml_simple(input_file):
    """
    Read OptoGait raw data from the excel xml file.
    The main information is stored in the spreadsheet "simple".

    Args:
        input_file (str): xml input file name.

    Returns:
        (DataFrame): DataFrame with information from the "simple" spreadsheet.
    """

    tree = ElementTree.parse(input_file)
    root = tree.getroot()

    rows = root.findall(
        "./{urn:schemas-microsoft-com:office:spreadsheet}Worksheet\
            [@{urn:schemas-microsoft-com:office:spreadsheet}Name='Simple']\
          /{urn:schemas-microsoft-com:office:spreadsheet}Table/"
    )
    header = [
        data.text
        for data in rows[0].findall(
            "./{urn:schemas-microsoft-com:office:spreadsheet}Cell/{urn:schemas-microsoft-com:office:spreadsheet}Data"
        )
    ]

    data = {}
    for column in header:
        data[column] = []

    for row_index, row in enumerate(rows[1:]):
        cell_index = 0
        for cell in row.findall("./{urn:schemas-microsoft-com:office:spreadsheet}Cell"):
            if "{urn:schemas-microsoft-com:office:spreadsheet}Index" in cell.attrib:
                jump_index = int(
                    cell.attrib["{urn:schemas-microsoft-com:office:spreadsheet}Index"]
                )
                while cell_index < jump_index - 1:
                    data[header[cell_index]].append(None)
                    cell_index += 1

            content = cell.find("./{urn:schemas-microsoft-com:office:spreadsheet}Data")
            while len(data[header[cell_index]]) < row_index:
                data[header[cell_index]].append(None)

            data[header[cell_index]].append(content.text)
            cell_index += 1

    return pd.DataFrame(data=data)
