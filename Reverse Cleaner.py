import os
import shutil


def reverse_organization(base_path):
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                source_path = os.path.join(folder_path, file_name)
                destination_path = os.path.join(base_path, file_name)
                shutil.move(source_path, destination_path)
            os.rmdir(folder_path)


if __name__ == "__main__":
    folder_path = "C:/Users/rodri/Downloads"
    reverse_organization(folder_path)
    print("Files reverted successfully.")