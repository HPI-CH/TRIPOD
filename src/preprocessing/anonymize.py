""" This script anonymizes raw data by removing sensitive data """

import os
import fnmatch

if __name__ == "__main__":  # noqa: C901
    directory = "./data/raw/TRIPOD"

    for root, d_names, f_names in os.walk(directory):
        for f in f_names:
            file_name = os.path.join(root, f)
            if fnmatch.fnmatch(f, "exp_?.mp4"):
                # remove non-blured video
                os.remove(file_name)
            elif fnmatch.fnmatch(f, "exp_?_blur.mp4"):
                # rename blured video
                try:
                    os.mkdir(os.path.join(root, "videos"))
                except FileExistsError:
                    pass
                os.rename(file_name, os.path.join(root, "videos", "side_view.mp4"))
            elif fnmatch.fnmatch(f, "*.asf"):
                # rename frontview
                try:
                    os.mkdir(os.path.join(root, "videos"))
                except FileExistsError:
                    pass
                os.rename(file_name, os.path.join(root, "videos", "front_view.asf"))
            elif fnmatch.fnmatch(f, "*.BIN"):
                # remove binary files and folder
                os.remove(file_name)
                if not os.listdir(root):
                    os.rmdir(root)
            elif fnmatch.fnmatch(f, "*.CSV"):
                # copy csv files to parent folder and remove date
                csv_file = open(file_name, "r")
                lines = csv_file.readlines()
                lines[1] = "Created on: YYYY-MM-DD hh:mm:ss\n"
                new_csv_file = open(os.path.join(root, os.pardir, f), "w")
                new_csv_file.writelines(lines)
                new_csv_file.close()
                csv_file.close()
                os.remove(file_name)
                if not os.listdir(root):
                    os.rmdir(root)
            elif fnmatch.fnmatch(root, "*OptoGait"):
                # remove non raw optogait files
                if (len(os.listdir(root)) == 2) and not fnmatch.fnmatch(f, "*_raw.xml"):
                    os.remove(file_name)
                else:
                    os.rename(file_name, os.path.join(root, "optogait_raw.xml"))
            elif fnmatch.fnmatch(root, "*Zebris"):
                if fnmatch.fnmatch(f, "*_raw.xml"):
                    os.rename(file_name, os.path.join(root, "zebris_raw.xml"))
                else:
                    os.rename(file_name, os.path.join(root, "zebris_steps.xml"))
