import glob
import os
import re

PATH = "logs/"

list_of_files = glob.glob(f"{PATH}*")
latest_file = max(list_of_files, key=os.path.getctime)
sorted_files = sorted(list_of_files, key=lambda x: int(x.split("/")[-1].split(".")[-2]), reverse=True)
top_two_files = sorted_files[:2]

# Select the file with "jupyter" in name, otherwise use the first one
file_with_largest_no = [f for f in top_two_files if "jupyter" in f][0]

print_next = False
with open(file_with_largest_no) as file:
    for line in file:
        if print_next:
            print(line.replace("tunnelling instructions:", "").strip())
            exit(0)
        if "tunnelling instructions" in line:
            print_next = True
