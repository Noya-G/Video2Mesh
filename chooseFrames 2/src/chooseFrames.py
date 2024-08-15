import logging
import math
import os
import re
import cv2
import sys
import time
import numpy as np
from LogMannager import create_new_folder
from eanalyzeTool import *
import subprocess
from concurrent.futures import ThreadPoolExecutor
from icecream import ic
from src.analysisTools.keypointCented import keypoint_movement_towards_center
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class CLIsetUP:
    __skip : int
    __threshhold : int
    __debug_mood : bool
    __root_output_location : str
    # __videos_paths : List[str] = []
    def __init__(self, skip = 10, threshold = 20, Debug_mood = False):
        self.__skip = skip
        self.__threshold = threshold
        self.__debug_mood = Debug_mood
        self.__videos_paths = []


    def add_video_path(self, path):
        if (not path) is str:
            raise ValueError("video path must be a str")
            return False
        elif not (os.path.exists(path)):
            raise ValueError("video path dosent exists")
            return False
        else:
            self.__videos_paths.append(path)
            if len(self.__videos_paths)==1:
                self.__root_output_location = self.__set_root_default_path()

            return True

    def __set_root_default_path(self):
        first_video_path = self.__videos_paths[0]
        path_as_array = first_video_path.split("/")
        path_as_array_without_0_len_str = [s for s in path_as_array if len(s) > 0]
        folder_back_path = path_as_array_without_0_len_str[:-1]
        root_output_location = '/' +os.path.join(*folder_back_path)
        return self.__create_firs_avalilable_path(root_output_location)

    def set_root_output_location(self,path:str):
        self.__root_output_location = path

    def get_root_output_location(self):
        return self.__root_output_location

    def get_video_paths(self):
        return self.__videos_paths

    def set_skip(self,skip):
        self.__skip = skip

    def get_skip(self):
        return self.__skip

    def set_threshold(self,threshhold):
        self.__threshhold = threshhold

    def get_threshold(self):
        return self.__threshold

    def set_debug_mood(self, Debug_mood):
        self.Debug_mood = Debug_mood

    def get_debug_mood(self):
        return self.Debug_mood

    def get_root_output_location(self):
        if self.__root_output_location == None:
            raise Exception("root output location is not set")
        return self.__root_output_location

    def set_root_output_location(self,root_output_location):
        self.__root_output_location = root_output_location

    def __create_firs_avalilable_path(self, path:str):
        itaration = 0
        potentioal_path = path+"/"+f"iteration_{itaration}"
        while os.path.exists(potentioal_path):
            itaration += 1
            potentioal_path = path + "/" + f"iteration_{itaration}"
        os.makedirs(potentioal_path)
        return  potentioal_path




SKIP = 20 # Number of frames to skip
THRESHOLD = 10
def extract_frames(video_path, skip=1):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_count = 0

    for i in tqdm(range(0, total_frames),
                       total=total_frames,
                       desc="Extracting Frames", ncols=75):
        success, frame = video_capture.read()
        if not success:
            break
        if i % SKIP == 0:
            frames.append(frame)

    # Release the video capture object
    video_capture.release()
    # print(f"extract {len(frames)} frames")


    return frames

def estimate_camera_movement(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Feature detection and matching (using SIFT)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match keypoints between the frames
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Estimate transformation (homography)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Decompose transformation
    dx = H[0, 2]  # Translation in x direction
    dy = H[1, 2]  # Translation in y direction
    theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi  # Rotation angle (in degrees)

    # Calculate camera movement
    translation_distance = np.sqrt(dx ** 2 + dy ** 2)

    return translation_distance, theta


def movement_estimator(frames):
    estimator = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in tqdm(range(0, len(frames)-1),
                      desc="movement estimator 1/2 processing",
                      ascii=False, ncols=100):
            frame1 = frames[i]
            frame2 = frames[i+1]
            future = executor.submit(estimate_camera_movement, frame1, frame2)
            futures.append((i, i +1, future))  # Store the frame indexes along with the future object

        # Retrieve results
        for start_index, end_index, future in tqdm(futures,
                                                   desc="movement estimator 2/2 processing",
                                                   ascii=False, ncols=100):
            translation_distance, theta = future.result()
            estimator.append((start_index, end_index, translation_distance, theta))  # Include the frame indexes in the results
        print(f"movement estimator finish successfully: {translation_distance}, theta: {theta}")
    return estimator


def run_cloudcompare(path1: str, path2: str):
    command_str = (
        "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare -SILENT \\\n"
        f"-O {path1} \\\n"
        f"-O {path2} \\\n"
        "-ICP -MIN_ERROR_DIFF 2 -ITER 200 -OVERLAP 35 \\\n"
        "-MERGE_MESHES"
    )

    # Print the command for verification
    print(f"Running command:\n{command_str}")

    # Replace newlines with spaces for the actual subprocess call
    command = command_str.replace("\\\n", " ").split()

    # Start the process
    process = subprocess.Popen(command)

    print("CloudCompare command is running...")

    # Continuously check if the process is still running
    while process.poll() is None:
        print("Still processing...")
        time.sleep(1)  # Wait for 1 second before checking again

    # Check the exit code
    if process.returncode == 0:
        print("CloudCompare command executed successfully.")
    else:
        print(f"An error occurred. Exit code: {process.returncode}")


def calculate_significance(frames, indexes):
    significance = []
    for i in tqdm(range(len(frames) - 1),
                  desc="calculate significance moovments processing",
                  ascii=False, ncols=100):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        translation_distance, theta = estimate_camera_movement(frame1, frame2)
        significance.append((indexes[i], indexes[i + 1], translation_distance, theta))
        print(f"movement significance estimator finish successfull")
    return significance


def select_frames_with_dynamic_programming(significance, limit):
    n = len(significance)
    dp = [0] * n
    for i in range(n):
        dp[i] = max(dp[j] for j in range(i) if significance[i][1] - significance[j][1] > THRESHOLD) + 1

    # Find the index of the last frame in the selected sequence
    max_index = max(range(n), key=lambda x: dp[x])

    # Backtrack to find the indexes of the frames in the selected sequence
    selected_indexes = []
    while max_index >= 0 and len(selected_indexes) < limit:
        selected_indexes.append(significance[max_index][0])
        max_index = max(j for j in range(max_index) if dp[j] == dp[max_index] - 1 and significance[max_index][1] - significance[j][1] > THRESHOLD)

    # Return the selected frames with their indexes
    return [(index, significance[index][1]) for index in selected_indexes]



def select_frames_with_indexes(frames, indexes, s):
    significance = calculate_significance(frames, indexes)
    selected_frames_with_indexes = select_frames_with_dynamic_programming(significance, s)
    selected_indexes = [frame[0] for frame in selected_frames_with_indexes]
    selected_frames = [frames[indexes.index(index)] for index in selected_indexes]
    return selected_frames, selected_indexes


def select_frames(movement):
    selected_frames = []
    for start_index, end_index, translation_distance, theta in tqdm(movement,
                                                                    desc=f"select frames using THRESHOLD {THRESHOLD} processing",
                                                                    ascii=False, ncols=100):
        if (translation_distance >THRESHOLD
                or abs(theta) > THRESHOLD):
            selected_frames.append(start_index)
    print(f"select frames using THRESHOLD {THRESHOLD} fineshed processing sucssesfuly")
    return selected_frames


def get_selected_frames(selected_frame_indexes, frames):
    selected_frames = []
    for index in selected_frame_indexes:
        selected_frames.append(frames[index])
    return selected_frames


def save_frames_as_photos(frames, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each frame as an image file
    for i, frame in tqdm(enumerate(frames),
                         desc=f"saving frames",
                         ascii=False, ncols=100):
        filename = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(filename, frame)
    print(f"frame saved sucssefuly to {output_folder}")

def detect_drone_movement(selected_frames,indexes):
    frames =list(zip(indexes, selected_frames))
    frames_with_movement_indexes = []

    # Variables to track the drone's movement
    ascending = False
    turning_down = False

    for idx, frame in frames:
        # Convert frame to grayscale for easier processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect edges in the frame
        edges = cv2.Canny(gray_frame, 50, 150)

        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if drone is ascending (going up)
                if y2 > y1:
                    ascending = True
                else:
                    ascending = False

                # Check if drone is turning its camera down
                if x1 == x2:
                    turning_down = True

        # If the drone has finished ascending and turned its camera down
        if ascending and turning_down:
            frames_with_movement_indexes.append(idx)

    return frames_with_movement_indexes


def configure_logging(log_dir):
    log_file_path = os.path.join(log_dir, "chosen_frames.log")
    error_log_file_path = os.path.join(log_dir, "error.log")

    # Configure logging to write INFO-level messages to chosen_frames.log
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Configure logging to write ERROR-level messages to error.log
    error_log_handler = logging.FileHandler(error_log_file_path)
    error_log_handler.setLevel(logging.ERROR)
    error_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    error_log_handler.setFormatter(error_log_formatter)
    logging.getLogger().addHandler(error_log_handler)

def get_git_branch():
    try:
        # Run git command to get current branch
        result = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        # Decode bytes to string and remove whitespace
        branch = result.decode('utf-8').strip()
        return branch
    except Exception as e:
        print(f"Error getting git branch: {e}")
        return None



def get_git_commit_info():
    try:
        # Get commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        # Get commit time
        commit_time = subprocess.check_output(['git', 'show', '-s', '--format=%ci', commit_hash]).decode('utf-8').strip()
        # Get commit message
        commit_message = subprocess.check_output(['git', 'show', '-s', '--format=%s', commit_hash]).decode('utf-8').strip()

        return commit_time, commit_message
    except Exception as e:
        print(f"Error getting git commit information: {e}")
        return None, None


def parse_log_file(log_file_path):
    import os

    # Check if the log file exists
    if not os.path.exists(log_file_path):
        # If the file doesn't exist, return default values
        return None, None, None, None, None

    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Extract relevant information using regular expressions
    skip_match = re.search(r'SKIP: (\d+)', log_content)
    threshold_match = re.search(r'THRESHOLD: (\d+)', log_content)
    branch_match = re.search(r'Branch: (\w+)', log_content)
    estimator_match = re.findall(r'movement estimator: \[(.*?)\]', log_content, re.DOTALL)
    frames_direction_match = re.findall(r'frames direction detected: \[(.*?)\]', log_content, re.DOTALL)

    if skip_match and threshold_match and branch_match:
        skip = int(skip_match.group(1))
        threshold = int(threshold_match.group(1))
        branch = branch_match.group(1)
        # Parse the estimator list if it exists
        if estimator_match:
            estimator_data = estimator_match[0]
            # Split the string by parentheses and extract numeric values
            estimator = [
                (int(data.split(',')[0]), int(data.split(',')[1]), float(data.split(',')[2]), float(data.split(',')[3]))
                for data in re.findall(r'\((.*?)\)', estimator_data)]
        else:
            estimator = None

        # Parse the frames_direction data if it exists
        if frames_direction_match:
            frames_direction_data = frames_direction_match[0]
            # Split the string by parentheses and extract numeric values
            frames_direction = [(int(data.split(',')[0]), int(data.split(',')[1]), int(data.split(',')[2])) for data in
                                re.findall(r'\((.*?)\)', frames_direction_data)]
        else:
            frames_direction = None

        return skip, threshold, branch, estimator, frames_direction
    else:
        return None, None, None, None, None


def movement_direction(frames, indexes):
    frames_direction = []
    framesWI = list(zip(indexes, frames))
    for i in tqdm(range(len(framesWI) - 1), # Iterate over the range of indices
                  desc=f"movement direction processing",
                  ascii=False, ncols=100):
        index1, frame1 = framesWI[i]
        index2, frame2 = framesWI[i + 1]
        direc = keypoint_movement_towards_center(frame1, frame2)
        frames_direction.append((index1, index2, direc))
    return frames_direction

def longest_subarray_with_value(lst):

    if not lst:
        return []

    n = len(lst)
    dp = [0] * n
    max_length = 0
    end_index = 0

    for i in range(n):
        if lst[i][2] == 4:
            dp[i] = dp[i - 1] + 1 if i > 0 else 1
            if dp[i] > max_length:
                max_length = dp[i]
                end_index = i

    start_index = end_index - max_length + 1
    return lst[start_index:end_index + 1]

def find_most_significant_indexes(data, size):
    n = len(data)

    # Create a table to store the maximum change for each index
    max_change_table = [0] * n

    # Calculate the maximum change for each index
    for i in range(n - 1, -1, -1):
        max_change = 0
        for j in range(i + 1, min(i + size + 1, n)):
            change = abs(data[j][2] - data[i][2]) + abs(data[j][3] - data[i][3])
            max_change = max(max_change, change)
        max_change_table[i] = max_change

    # Backtrack to find the indexes with the most significant changes
    significant_indexes = []
    current_index = 0
    while len(significant_indexes) < size:
        significant_indexes.append(data[current_index][0])  # Append the first element of the chosen tuple
        next_index = current_index + 1
        for i in range(current_index + 1, min(current_index + size + 1, n)):
            if max_change_table[i] > max_change_table[next_index]:
                next_index = i
        current_index = next_index

    return significant_indexes


def largest_interval_subset(original_list, size=100):
    if len(original_list) <= 100:
        return original_list

    ic(len(original_list))
    ic(size)
    step = len(original_list) // size  # Calculate step size for subset

    if step == 0:  # If original list is smaller than size 100
        return original_list

    subset = []
    for i in range(0, len(original_list), step):
        subset.append(original_list[i])

    # Adjust the size of the subset to be exactly 100
    if len(subset) < size:
        last_index = len(original_list) - 1
        while len(subset) < size and last_index >= 0:
            subset.append(original_list[last_index])
            last_index -= step

    return subset[:size]

import subprocess
import logging

def create_mesh_by_ODM(path):
    command = [
        "docker", "run", "--rm",
        "-v", f"{path}:/datasets",
        "opendronemap/odm", "--project-path", "/datasets", "project"
    ]

    try:
        # Use Popen to start the process
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print the output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                logging.debug(output.strip())

        # Capture the remaining stderr
        stderr = process.stderr.read()
        if stderr:
            print(stderr.strip(), file=sys.stderr)
            logging.error(stderr.strip())

        # Check the return code
        rc = process.poll()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, command)

    except subprocess.CalledProcessError as e:
        logging.error("Command '%s' returned non-zero exit status %s.", e.cmd, e.returncode)
    except Exception as e:
        logging.exception("An error occurred while running the command.")

# def create_mesh_by_ODM(path):
#     command = [
#         "docker", "run", "--rm",
#         "-v", f"{path}:/datasets",
#         "opendronemap/odm", "--project-path", "/datasets", "project"
#     ]
#
#     try:
#         # Run the command
#         result = subprocess.run(command, capture_output=True, text=True, check=True)
#
#         # Print and log the output
#         print(result.stdout)
#         print(result.stderr, file=sys.stderr)
#         logging.debug("STDOUT: %s", result.stdout)
#         logging.error("STDERR: %s", result.stderr)
#
#     except subprocess.CalledProcessError as e:
#         logging.error("Command '%s' returned non-zero exit status %s.", e.cmd, e.returncode)
#         logging.error("STDOUT: %s", e.stdout)
#         logging.error("STDERR: %s", e.stderr)
#     except Exception as e:
#         logging.exception("An error occurred while running the command.")

# Example usage (make sure to set this path accordingly)
# create_mesh_by_ODM("/path/to/your/dataset")

def select_frames_with_most_significant_movements(selected_frames, selected_frames_indexes, limit):
    # Calculate significance of movements
    significance = calculate_significance(selected_frames, selected_frames_indexes)

    # Select frames with the most significant movements using dynamic programming
    selected_frames_with_indexes = select_frames_with_dynamic_programming(significance, limit)

    # Extract selected frames and indexes
    selected_indexes = [frame[0] for frame in selected_frames_with_indexes]
    selected_frames = [selected_frames[selected_frames_indexes.index(index)] for index in selected_indexes]

    return selected_frames, selected_indexes

def get_log_file(cond,video_path, destPath,format_type = "png" ,
                 rate=SKIP,count=100,Limit=100,precent =100, preLog=False):
    # output_payh =
    video_name = video_path.split("/")[-1]
    SKIP = rate
    video_path = video_path
    video_name = last_six_chars = video_path[-6:]
    directory = os.path.dirname(video_path)
    new_path = destPath
    if cond is True:
        log_file_path = os.path.join(destPath, "chosen_frames.log")
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logERORR_file_path = os.path.join(new_path, 'error.log')
        logging.basicConfig(filename=logERORR_file_path, level=logging.ERROR)

        # Get current branch
        current_branch = get_git_branch()

        # Get commit information
        commit_time, commit_message = get_git_commit_info()

        if commit_time and commit_message:
            logging.info(f"Commit Time: {commit_time}")
            logging.info(f"Commit: {commit_message}")
        else:
            logging.info("Commit information not available")

        if current_branch:
            logging.info(f"Branch: {current_branch}")
        else:
            logging.info("Branch information not available")
        logging.info(f"SKIP: {SKIP}")
        logging.info(f"THRESHOLD: {THRESHOLD}")
        logging.info(f"Video name: {video_name}")

    # Extract frames from the video
    all_frames = extract_frames(video_path)
    if cond is True:
        logging.info(f"Total frames extracted: {len(all_frames)}")

        # Load the previous log file
        if preLog is True:
            previous_log_file_path = '/Users/noyagendelman/Desktop/choosingFrames/chosen_frames_21/chosen_frames.log'  # Change this to the path of your previous log file
            previous_skip, previous_threshold, previous_branch, previous_estimator, previous_frames_direction = parse_log_file(
                previous_log_file_path)

            # Check if any previous parameter is None
            if any(param is None for param in
                   [previous_skip, previous_threshold, previous_branch, previous_estimator, previous_frames_direction]):
                # Execute the method to proceed with regular execution
                estimator = movement_estimator(all_frames)
                frames_indexes = select_frames(estimator)

                frames_direction = movement_direction(all_frames, frames_indexes)
            else:
                # Proceed with regular execution
                estimator = previous_estimator
                frames_direction = previous_frames_direction

        else:
            estimator = movement_estimator(all_frames)
            frames_indexes = select_frames(estimator)
            frames_direction = movement_direction(all_frames, frames_indexes)

        logging.info("Using previous estimator and frames_direction from the log file.")
        # Select frames with significant camera movement
        logging.info(f"Total pairs of frames processed: {len(estimator)}")
        logging.info(f"movement estimator: {estimator}")
        logging.info(f"frames direction detected: {frames_direction}")


    if cond is False:
        estimator = movement_estimator(all_frames)
        # Select frames with significant camera movement
        selected_frames_indexes = select_frames(estimator)
        frames_indexes = select_frames(estimator)
        frames_direction = movement_direction(all_frames, frames_indexes)

    if cond is True:
        # estimator = movement_estimator(all_frames)
        selected_frames_indexes = select_frames(estimator)
        print(selected_frames_indexes)
        if selected_frames_indexes is None:
            selected_frames_indexes = select_frames(estimator)
        logging.info(f"Total selected frames: {len(selected_frames_indexes)}")
        logging.info(f"selected frames: {selected_frames_indexes}")

    # Get selected frames
    selected_frames = get_selected_frames(selected_frames_indexes, all_frames)

    # Save selected frames as images
    # save_frames_as_photos(selected_frames, new_path)
    expected_translation_distance, expected_theta = calculate_expected_values(estimator,
                                                                              selected_frames,
                                                                              selected_frames_indexes)
    if cond is True:
        logging.info(f"expected translation distance: {expected_translation_distance}")
        logging.info(f"expected theta: {expected_theta}")
        # Plot translation distance
        plot_translation_distance(expected_translation_distance, estimator[:len(selected_frames_indexes) - 1],
                                  selected_frames_indexes, new_path)

        # Plot theta
        plot_theta(expected_theta, estimator[:len(selected_frames_indexes) - 1], selected_frames_indexes,
                   new_path)
        plot_theta_zoom_out(expected_theta, estimator[:len(selected_frames_indexes) - 1], selected_frames_indexes,
                            new_path)

    longestSubarray = longest_subarray_with_value(frames_direction)

    firstGoodFrame = longestSubarray[-1][1]
    if (firstGoodFrame < selected_frames_indexes[math.floor(len(selected_frames_indexes) / 2)]):
        selected_frames_indexes = selected_frames_indexes[selected_frames_indexes.index(firstGoodFrame):]
        if cond is True:
            logging.info(f"selected frames after cutting: {selected_frames_indexes}")
            logging.info(f"total selected frames after cutting: {len(selected_frames_indexes)}")
        selected_frames = get_selected_frames(selected_frames_indexes, all_frames)

    if (firstGoodFrame > selected_frames_indexes[math.floor(len(selected_frames_indexes) / 2)]):
        selected_frames_indexes = selected_frames_indexes[:selected_frames_indexes.index(firstGoodFrame)]
        if cond is True:
            logging.debug(f"selected frames after cutting: {selected_frames_indexes}")
            logging.debug(f"total selected frames after cutting: {len(selected_frames_indexes)}")
        selected_frames = get_selected_frames(selected_frames_indexes, all_frames)

    # estimator2 = movement_estimator(selected_frames)
    selected_indexes = largest_interval_subset(selected_frames_indexes,count)
    if cond is True:
        logging.info(f"final selected frames indexes: {selected_frames_indexes}")
    selected_frames = get_selected_frames(selected_indexes, all_frames)
    final_path_target = f"{destPath}/project/images"
    save_frames_as_photos(selected_frames, final_path_target)
    logging.info("start building mesh using ODM")
    logging.debug(f"ODM using the path {destPath}/project")
    logging.shutdown()
    # Define the command





