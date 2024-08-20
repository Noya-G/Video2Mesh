import os

SKIP = 20

def create_new_folder(base_path, folder_name):
    """
    Create a new folder with a unique name in the specified base path.

    Args:
        base_path (str): The path where the new folder will be created.
        folder_name (str): The desired name for the new folder.

    Returns:
        str: The path of the newly created folder.
    """
    # Check if the base path exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path '{base_path}' does not exist.")

    # Ensure folder name is valid
    folder_name = folder_name.strip()

    # Iterate to find a unique folder name
    new_folder_name = folder_name
    counter = 1
    while os.path.exists(os.path.join(base_path, new_folder_name)):
        new_folder_name = f"{folder_name}_{counter}"
        counter += 1

    # Create the new folder
    new_folder_path = os.path.join(base_path, new_folder_name)
    os.makedirs(new_folder_path)

    return new_folder_path
