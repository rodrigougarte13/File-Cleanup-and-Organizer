import os
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# extract info about the files
def extract_file_info(folder_path):  # returns for each file: [file_name, file_type, file_size, creation_date]
    info_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_type = os.path.splitext(file_path)[1][1:]
            file_size = os.path.getsize(file_path) / 1024
            creation_time = os.path.getctime(file_path)
            creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
            info_list.append([file_name, file_type, file_size, creation_date])
    return info_list


def organize_files_by_type(folder_path, info_list):
    for file_info in info_list:
        file_name = file_info[0]
        file_type = file_info[1]
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_path}")  # Debug print
        if os.path.isfile(file_path):
            type_folder_path = os.path.join(folder_path, file_type)
            target_path = os.path.join(type_folder_path, file_name)
            if not os.path.exists(type_folder_path):
                os.makedirs(type_folder_path)
                print(f"Created directory: {type_folder_path}")  # Debug print
            shutil.move(file_path, target_path)
            print(f"Moved {file_path} to {target_path}")


if __name__ == "__main__":
    folder_path = "C:/Users/rodri/Downloads"
    file_info = extract_file_info(folder_path)
    organize_files_by_type(folder_path, file_info)
    print("Files organized successfully.")