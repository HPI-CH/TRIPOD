""" This script converts all excel xml files from OptoGait to CSV files.
    Attention: Files are replaced (XML files are deleted) """

import os
import fnmatch
from src.data_reader.opto_gait_xml_reader import read_opto_gait_raw_xml_simple

if __name__ == "__main__":
    directory = "./data/raw/TRIPOD"

    for root, d_names, f_names in os.walk(directory):
        for f in f_names:
            file_name = os.path.join(root, f)
            if fnmatch.fnmatch(f, "optogait_raw.xml"):
                df = read_opto_gait_raw_xml_simple(file_name)

                drop_columns = df.isna().all()
                drop_columns[
                    [
                        "Last name",
                        "Name",
                        "Last & first names",
                        "BirthDate",
                        "Gender",
                        "Test",
                        "Date",
                        "Time",
                    ]
                ] = True

                df.drop(columns=df.columns[drop_columns], inplace=True)

                df.to_csv(os.path.join(root, "optogait.csv"))

                os.remove(file_name)
