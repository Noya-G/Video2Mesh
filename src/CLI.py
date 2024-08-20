# import os
# import signal
# import sys
# from colorama import Fore, Style, init
# from src import UserInput
# from src.chooseFrames import get_log_file
#
# # Import the frame_maker module for frame extraction
# import pyfiglet
# signal.signal(signal.SIGINT, signal.SIG_DFL)  # Reset SIGINT signal handling to default
#
# # Help message for the CLI
# help_message = \
#     "\nVideo2ImageCLI - Image Extraction Tool\n\n\
#     ===============================================================\n\n\
#     Usage:\n\
#       [command] [options]\n\
#     Commands:\n\
#       extract           Extract frames from a video file.\n\
#       help              Show this help message.\n\n\
#     Options:\n\
#       -v, --video       Path to the video file (required).\n\
#       -n  --numberOfVideos  Number of videos to extract.\n\
#       -o, --output      Output directory for extracted images. Default is 'images'.\n\
#       -f, --format      Image format (e.g., png, jpg). Default is 'png'.\n\
#       -r, --rate        Frame extraction rate (e.g., 1 frame every 5 seconds). Default is 5.\n\
#       -c, --count       Extract frames by quantity (will be averaged by length).\n\
#       -l, --limit       Limit the number of frames to be extracted.\n\
#       -p, --percent     Extract frames by percentage difference (e.g., 10 - above this percentage difference).\n\
#       -?, --help        Display this help message.\n\n\
#     Examples:\n\
#       # Extract one frame every 5 seconds from a video and save as PNG\n\
#       extract --video=myvideo.mp4 --output=images --rate=5 --format=png\n\
#       # Extract 100 frames from video and save them in the 'images' directory as PNG files\n\
#       extract -v=myvideo.mp4 -o=images -c=100 -f=png\n\
#       # Extract one frame every 1 second from a video and save as PNG files in 'images' directory.\n\
#         Additionally, use in-memory mode to take a frame only if there is a change above 10% with a total limitation of 50 frames.\n\
#       extract -v=video.mp4 -o=images -f=png -r=1 -m=True -p=10 -l=50\n\
#     Notes:\n\
#       - Ensure the video file path is correct.\n\
#       - Make sure the output directory is writable.\n\
#       - The tool requires appropriate permissions to read and write files.\n\n\
#     For more information, visit our documentation at [link to documentation].\n\n\
#     ---\n\n\
#     Feel free to adjust command names, options, and examples to align with your CLI tool's capabilities and design.\n\
#     "
#
# # Welcome message for the CLI
# welcome_message = "\n\n" \
#                   "This is a command-line tool to process images from videos.\n" \
#                   "Simply follow the prompts or use the --help option to learn more about the available commands.\n\n"
#
# # Error message for invalid commands
# error_message = "Invalid command. Please try again or use 'help'."
#
# # Prompt for the CLI
# cli_message = "Video2ImageCLI>>> "
#
# # List of flags for help and exit
# help_flags = ['help', '--help', '-help', '-h', '-?']
# exit_flags = ['quit()', 'quit', 'exit']
#
# # List of flags for extraction command
# extract_flags = ['-v', '--video', '-o', '--output', '-f', '--format', '-r', '--rate', '-m', '--memory',
#                  '-c', '--count', '-l', '--limit', '-p', '--percent']
#
# # List of flags requiring integer values
# int_val = ['-r', '-c', '-l', '-p']
#
# # List of flags requiring boolean values
# bool_val = ['-m']
#
#
# def parse_flag(flag_lst):
#     """
#     Parse command-line flag and its value.
#     """
#     if flag_lst[0] not in extract_flags:
#         print(error_message)
#         return 0, None
#     reduce_flag = flag_lst[0].replace('-', '', 1) if len(flag_lst[0]) > 2 else flag_lst[0]
#     key = reduce_flag[:2]
#     value = flag_lst[1]
#     if key in int_val:
#         try:
#             value = int(value)
#         except ValueError as v_e:
#             print(f"Invalid command. Check - {v_e}")
#             value = None
#     elif key in bool_val:
#         if value in ['False', 'True']:
#             value = bool(value)
#         else:
#             print(f"Expect to receive boolean value ['False', 'True'] but got - {value}")
#             value = None
#     return key, value
#
#
# def parse_command(command_p):
#     """
#     Parse command-line command and its options.
#     """
#     extract_option = {'-o': 'images', '-f': 'png', '-r': 5, '-m': False}
#     for option in command_p:
#         opt_k_v = option.split('=')
#         if len(opt_k_v) == 2:
#             key, value = parse_flag(opt_k_v)
#             if value is None:
#                 return
#             extract_option[key] = value
#         else:
#             print(f"{error_message}. Check - {option}")
#             return
#     if extract_option.get('-v') is None:
#         print("Need to insert path to input file, try again.")
#         return
#     if len(extract_option) > 0:
#         return extract_option
#     return
#
#
# def command_management(command):
#     """
#     Manage the execution of commands provided by the user.
#     """
#     video_path = command.get('-v')
#     if not os.path.isfile(video_path):
#         print(f"The file does not exist. Please enter the correct path and try again.")
#         return
#     output_folder = command.get('-o', 'images')
#     if not os.path.isdir(output_folder):
#         os.makedirs(output_folder)
#     format_type = command.get('-f', 'png')
#     rate = command.get('-r', 5)
#     memory = command.get('-m', False)
#     count = command.get('-c',100)
#     limit = command.get('-l')
#     percent = command.get('-p')
#
    # for video in video_path:
    #     # Call the frame extraction function from your tool
    #     get_log_file(True, video, output_folder, format_type,
    #                  rate, count, limit, percent)
#
#     print("Frames extraction completed successfully.")
#
#
#
# if __name__ == '__main__':
#     UserInput.cli_engine()
#
