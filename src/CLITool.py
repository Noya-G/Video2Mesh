import imghdr
import os
import shutil
import subprocess
import sys

import trimesh
import signal

import pyfiglet
from colorama import Fore, init

# from src.CLI import welcome_message, cli_message, exit_flags, help_flags, help_message, error_message, extract_flags, \
#     int_val, bool_val
from chooseFrames import get_log_file, CLIsetUP, create_mesh_by_ODM, run_cloudcompare

signal.signal(signal.SIGINT, signal.SIG_DFL)  # Reset SIGINT signal handling to default




def check_docker_running():
    try:
        # Try to run a simple Docker command to check if Docker is running
        result = subprocess.run(['docker', 'info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command was successful
        if result.returncode != 0:
            print("Error: Docker is not running. Please start Docker and try again.")
            return False
        else:
            print("Docker is running.")
            return True
    except FileNotFoundError:
        print("Error: Docker is not installed or not found in PATH.")
        return False

def start_docker_on_mac():
    try:
        subprocess.run(['open', '/Applications/Docker.app'], check=True)
        print("Starting Docker on macOS...")
    except subprocess.CalledProcessError:
        print("Failed to start Docker on macOS. Please start it manually.")

def start_docker_on_windows():
    try:
        subprocess.run(['start', '/B', 'Docker Desktop'], shell=True)
        print("Starting Docker on Windows...")
    except subprocess.CalledProcessError:
        print("Failed to start Docker on Windows. Please start it manually.")

def ensure_docker_running():
    if not check_docker_running():
        if sys.platform == "darwin":  # macOS
            start_docker_on_mac()
        elif sys.platform == "win32":  # Windows
            start_docker_on_windows()

        # After attempting to start Docker, check again
        if not check_docker_running():
            print("Docker is still not running. Exiting program.")
            sys.exit(1)


user_introduction_option = ""

def help():
    help_text = """
        Video 2 Mesh CLI Tool

        This command-line tool allows you to process images from videos and create 3D mesh models. Below are the available commands and their descriptions:

        Available Commands:

        -full_process: 
            Converts two video files into one mesh model. This process involves extracting frames from the videos,
            creating meshes for each video, and then merging them into a single mesh model.

        -video2images:
            Extracts the best frames from a video to create a mesh model. You can configure parameters like skip and threshold
            to control how frames are selected.

        -create_mesh:
            Creates a mesh model from a set of frames located in a specified directory. The directory structure should follow
            a specific format, where images are placed inside a 'project/images' folder.

        -merge:
            Merges two existing mesh models into one. You need to provide the paths to the two mesh models that you want to merge.

        Usage:
        1. Run the tool and select the desired command.
        2. Follow the prompts to input video paths, configure parameters, or provide mesh paths as needed.
        3. The tool will guide you through the process and display relevant information and progress.

        Examples:
        - To process two videos into a single mesh model:
          Enter: -full_process

        - To extract frames from a video and prepare them for mesh creation:
          Enter: -video2images

        - To create a mesh from existing frames:
          Enter: -create_mesh

        - To merge two mesh models:
          Enter: -merge

        Additional Notes:
        - Use valid paths when prompted. Ensure that the video files, images, and meshes are located in the specified directories.
        - If you wish to change default parameters like 'skip' and 'threshold', you can do so during the process.
        - For further assistance, refer to the comments in the source code or consult the project documentation.

        """
    print(help_text)
    print("Press Enter to continue to go back to the main menu.")
    input()
    main_menu()

def welcome_message():
    welcome_message = "This is a command-line tool to process images from videos.\n"\
                      "Simply follow the prompts or use the --help option to learn more about the available commands.\n\n"

    user_introduction_option = "Enter:\n"\
                               "-full_process       for converting 2 video files into one messh model.\n"\
                               "-video2images       for getting the best frames from the video to create mesh.\n"\
                               "-create_mesh        creat mesh model form sets of frames.\n"\
                               "-merge              to merge 2 meshes into one mesh.\n" \
                               "-help               to get help"


    init(autoreset=True)
    big_title = pyfiglet.figlet_format("Video 2 Mesh")
    print(Fore.BLUE + big_title)
    print(welcome_message)


def main_menu():
    user_introduction_option = "Enter:\n" \
                               "-full_process       for converting 2 video files into one messh model.\n" \
                               "-video2images       for getting the best frames from the video to create mesh.\n" \
                               "-create_mesh        creat mesh model form sets of frames.\n" \
                               "-merge              to merge 2 meshes into one mesh.\n"\
                               "-help               to get help\n"\
                               "-exit              exit video 2 mesh\n"
    user_input = input(user_introduction_option)
    # got_input = False
    while True:
        if user_input == ("-full_process" or "full_process"):
            print("full_process")
            full_process()
            break
        elif user_input == ("-video2images" or "video2images"):
            video2images()
            break
        elif user_input == ("-create_mesh" or "create_mesh"):
            create_mesh()
            break
        elif user_input == ("-merge" or "merge"):
            merge()
            break
        elif user_input == ("-help" or "help"):
            help()
            break
        elif user_input == "-exit":
            print("good bye.")
            sys.exit()
        print("incorrect input. try again:")
        user_input = input(user_introduction_option)



def full_process():
    setup = CLIsetUP()

    # Adding first video path
    video1_path = input("Enter the path of the first video: ")
    video1_path = convert_window_path(video1_path)
    adding_path1_successfully = setup.add_video_path(video1_path)

    if adding_path1_successfully:
        print(f"{video1_path} has been added successfully.")

    # Adding second video path
    video2_path = input("Enter the path of the second video: ")
    video2_path = convert_window_path(video2_path)
    adding_path2_successfully = setup.add_video_path(video2_path)
    if adding_path2_successfully:
        print(f"{video2_path} has been added successfully.")

    # Display default parameters
    skip = setup.get_skip()
    threshold = setup.get_threshold()
    dest_path = setup.get_root_output_location()
    print(f"The default parameters are:\n"
          f"skip = {skip}\n"
          f"threshold = {threshold}\n"
          f"output path = {dest_path}")

    # Asking the user if they want to change parameters
    is_user_want_to_change_parameters = input("\nEnter Y to change the parameters or N to keep them: ").strip().upper()

    if is_user_want_to_change_parameters == 'Y':
        # Change skip value
        new_skip = int(input(f"Enter the new value for skip (current: {skip}): "))
        setup.set_skip(new_skip)

        # Change threshold value
        new_threshold = int(input(f"Enter the new value for threshold (current: {threshold}): "))
        setup.set_threshold(new_threshold)

        new_output_path = input(f"Enter the new value for output loacation (current: {dest_path}): ")
        setup.set_root_output_location(new_output_path)

        print(f"Parameters updated: skip = {setup.get_skip()}, threshold = {setup.get_threshold()}")
    else:
        print("Keeping the default parameters.")

    print()
    paths = setup.get_video_paths()
    i = 1
    for video_path in paths:
        dest_path_per_video = dest_path+f"/v{i}"
        os.makedirs(dest_path_per_video)
        get_log_file(setup,True, video_path, dest_path_per_video, "png",
                     5, 100, 100, 100,False)
        create_mesh_by_ODM(dest_path_per_video)
        i += 1
    mesh1_path = dest_path + "/v1/project/odm_texturing_25d/odm_textured_model_geo.obj"
    mesh2_path = dest_path + "/v2/project/odm_texturing_25d/odm_textured_model_geo.obj"
    run_cloudcompare(mesh1_path,mesh2_path)
    # Placeholder for next steps
    print("Full process completed.")

def video2images():
    setup = CLIsetUP()

    # Adding first video path
    video1_path = input("Enter the path of the first video: ")
    video1_path = convert_window_path(video1_path)
    setup = CLIsetUP()
    adding_path1_successfully = setup.add_video_path(video1_path)

    if adding_path1_successfully:
        print(f"{video1_path} has been added successfully.")

        # Display default parameters
        skip = setup.get_skip()
        threshold = setup.get_threshold()
        dest_path = setup.get_root_output_location()
        print(f"The default parameters are:\n"
              f"skip = {skip}\n"
              f"threshold = {threshold}\n"
              f"output path = {dest_path}")

        is_user_want_to_change_parameters = input(
            "\nEnter Y to change the parameters or N to keep them: ").strip().upper()

        if is_user_want_to_change_parameters == 'Y':
            # Change skip value
            new_skip = int(input(f"Enter the new value for skip (current: {skip}): "))
            setup.set_skip(new_skip)

            # Change threshold value
            new_threshold = int(input(f"Enter the new value for threshold (current: {threshold}): "))
            setup.set_threshold(new_threshold)

            new_output_path = input(f"Enter the new value for output loacation (current: {dest_path}): ")
            setup.set_root_output_location(new_output_path)

            print(f"Parameters updated: skip = {setup.get_skip()}, threshold = {setup.get_threshold()}")
        else:
            print("Keeping the default parameters.")

        print()
        paths = setup.get_video_paths()
        i = 1
        for video_path in paths:
            dest_path_per_video = dest_path + f"/v{i}"
            os.makedirs(dest_path_per_video)
            get_log_file(setup,True, video_path, dest_path_per_video, "png",
                         5, 100, 100, 100, False)

def create_mesh():
    video1_path = input("Enter the path of images folder: ")
    adding_path1_successfully, directory_path = is_vaid_path_for_ODM(video1_path)
    if adding_path1_successfully:
        print(f"{video1_path} has been added successfully.")
    parent_directory = os.path.dirname(directory_path)
    parent2_directory = os.path.dirname(parent_directory)
    create_mesh_by_ODM(parent2_directory)

def merge():
    def __print_invalid_path():
        return print(f"the path you entered doest not exist. Please try again.")

    mesh_path1 = None
    mesh_path2 = None

    legal_path1 = False
    legal_path2 = False
    while not legal_path1:
        mesh_path1 = input("Enter the path of the first mesh: ")
        convert_window_path(mesh_path1)
        if os.path.exists(mesh_path1):
            legal_path1 = True
        else:
            __print_invalid_path()
    while not legal_path2:
        mesh_path2 = input("Enter the path of the second: ")
        convert_window_path(mesh_path2)
        if os.path.exists(mesh_path2):
            legal_path2 = True
        else:
            __print_invalid_path()

    run_cloudcompare(mesh_path1,mesh_path2)
def convert_window_path(directory_path: str):
    return directory_path.replace("\\","/")

def is_vaid_path_for_ODM(directory_path: str):
    original_directory_path = directory_path

    contain_images = path_contains_images(directory_path)
    if contain_images is False:
        print(f"{directory_path} does not contain images")

    project_folder =  directory_path.split("/")[-2]
    images_folder = directory_path.split("/")[-1]

    if project_folder != "project":
        directory_path = directory_path+"/project"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    if images_folder != "images":
        directory_path = directory_path+"/images"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    if original_directory_path != directory_path:
        move_images_between_directories(original_directory_path,directory_path)
    return True, directory_path


def move_images_between_directories(source_directory, destination_directory):
    """
    Moves all image files from the source directory to the destination directory.

    :param source_directory: The path to the directory containing the images to be moved.
    :param destination_directory: The path to the directory where the images will be moved.
    :return: None
    """
    # Validate the source directory path
    if not os.path.isdir(source_directory):
        raise ValueError(f"The path {source_directory} is not a valid directory.")

    # Validate or create the destination directory path
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Move image files from source to destination directory
    for filename in os.listdir(source_directory):
        source_file_path = os.path.join(source_directory, filename)

        if os.path.isfile(source_file_path) and imghdr.what(source_file_path):
            # Move the image file to the destination directory
            shutil.move(source_file_path, os.path.join(destination_directory, filename))

    print(f"All images have been moved from {source_directory} to {destination_directory}")


def path_contains_images(directory_path: str):
    """
    Checks if the specified directory contains any image files.

    :param directory_path: The path to the directory to check.
    :return: True if the directory contains at least one image file, otherwise False.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"The path {directory_path} is not a valid directory.")

        # Loop through all the files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            # Check if the file is an image
            if imghdr.what(file_path):
                return True
    print(f"The path {directory_path} doesnt contains any image files")
    return False

def cli_engine():
    while True:
        welcome_message()
        main_menu()
        print("")


if __name__ == '__main__':
    ensure_docker_running()

    cli_engine()

