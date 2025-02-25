import os

def rename_files_to_ts(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.endswith('.1'):
            new_filename = os.path.join(folder_path, filename.rsplit('.', 1)[0]+ '.mp4')
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(old_filepath, new_filepath)

# 示例用法
folder_path = r'D:\811'
rename_files_to_ts(folder_path)
