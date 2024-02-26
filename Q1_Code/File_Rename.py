import os

def rename_files(folder_path):
    # Get list of files in the folder
    files = os.listdir(folder_path)
    
    # Initialize counter for numbering files
    count = 1
    
    # Iterate over each file
    for file in files:
        # Check if the file is a regular file (not a directory)
        if os.path.isfile(os.path.join(folder_path, file)):
            # Get the file extension
            _, ext = os.path.splitext(file)
            
            # Rename the file
            new_name = f"synthetic_{count:03d}{ext}"
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
            
            # Increment counter
            count += 1

# Replace 'folder_path' with the path to your folder
folder_path = '/home/coder/workspace/Data/Synthetic_Data/'
rename_files(folder_path)

