# import os
# import sys
# import signal
#
# import pyfiglet
# from colorama import Fore, init
#
# from src.CLI import welcome_message, cli_message, exit_flags, help_flags, help_message, error_message, extract_flags, \
#     int_val, bool_val
# from src.chooseFrames import get_log_file
#
# signal.signal(signal.SIGINT, signal.SIG_DFL)  # Reset SIGINT signal handling to default
#
# class UserInput:
#     def __init__(self, command):
#         self._video_paths = [None, None]
#         self._output = 'images'
#         self._format = 'png'
#         self._rate = 5
#         self._count = 100
#         self._limit = None
#         self._percent = None
#         self.help = False
#         self.numberOfVideos=1
#         self.output_path = []
#
#
#
#         command_lst = command.split(' ')
#         if command_lst[0] == 'extract':
#             command_dict = self.parse_command(command_lst[1:])
#             if command_dict is not None:
#                 self.set_options(command_dict)
#                 self.command_management()
#         elif command_lst[0] in help_flags:
#             self.help = True
#             print(help_message)
#         else:
#             print(error_message)
#
#     def get_video_paths(self):
#         return self._video_paths
#
#     def set_video_paths(self, paths):
#         self.set_numberOfVideos(paths)
#         for i, path in enumerate(paths):
#             if not os.path.isfile(path):
#                 print(f"The file does not exist: {path}. Please enter the correct path and try again.")
#                 return False
#             self._video_paths[i] = path
#         return True
#
#
#     def set_numberOfVideos(self, paths):
#         self.numberOfVideos = len(paths)
#     def get_output(self):
#         return self._output
#
#     def set_output(self, output_folder):
#
#         path_end = self._video_paths[0].split("/")[-1]
#         path_name = path_end.split(".")[0]
#         suffix = 0
#         output_path = f"{output_folder}/iteration"
#
#         # Check if the directory exists, and increment the suffix until we find a unique name
#         unique_output_path = f"{output_path}_{suffix}"
#         while os.path.exists(unique_output_path):
#             unique_output_path = f"{output_path}_{suffix}"
#             suffix += 1
#
#         # Create the unique directory
#         os.makedirs(unique_output_path)
#         self.output_path.append(f"{unique_output_path}/v1")
#         os.makedirs(f"{unique_output_path}/v1")
#         self.output_path.append(f"{unique_output_path}/v2")
#         os.makedirs(f"{unique_output_path}/v2")
#
#
#     def get_format(self):
#         return self._format
#
#     def set_format(self, format_type):
#         self._format = format_type
#
#     def get_rate(self):
#         return self._rate
#
#     def set_rate(self, rate):
#         self._rate = rate
#
#     def get_count(self):
#         return self._count
#
#     def set_count(self, count):
#         self._count = count
#
#     def get_limit(self):
#         return self._limit
#
#     def set_limit(self, limit):
#         self._limit = limit
#
#     def get_percent(self):
#         return self._percent
#
#     def set_percent(self, percent):
#         self._percent = percent
#
#     def parse_flag(self, flag_lst):
#         if flag_lst[0] not in extract_flags:
#             print(error_message)
#             return None, None
#         reduce_flag = flag_lst[0].replace('-', '', 1) if len(flag_lst[0]) > 2 else flag_lst[0]
#         key = reduce_flag[:2]
#         value = flag_lst[1]
#         if key in int_val:
#             try:
#                 value = int(value)
#             except ValueError as v_e:
#                 print(f"Invalid command. Check - {v_e}")
#                 value = None
#         elif key in bool_val:
#             if value in ['False', 'True']:
#                 value = value == 'True'
#             else:
#                 print(f"Expect to receive boolean value ['False', 'True'] but got - {value}")
#                 value = None
#         return key, value
#
#     def parse_command(self, command_p):
#         extract_option = {'-o': 'images', '-f': 'png', '-r': 5, '-m': False}
#         video_paths = []
#         for option in command_p:
#             opt_k_v = option.split('=')
#             if len(opt_k_v) == 2:
#                 if opt_k_v[0] in ['-v1', '--video1', '-v2', '--video2']:
#                     video_paths.append(opt_k_v[1])
#                 else:
#                     key, value = self.parse_flag(opt_k_v)
#                     if value is None:
#                         return None
#                     extract_option[key] = value
#             else:
#                 print(f"{error_message}. Check - {option}")
#                 return None
#         if len(video_paths) != 2:
#             print("You must provide paths for two video files.")
#             return None
#         if not self.set_video_paths(video_paths):
#             return None
#         return extract_option
#
#     def set_options(self, command):
#         self.set_output(command.get('-o', 'images'))
#         self.set_format(command.get('-f', 'png'))
#         self.set_rate(command.get('-r', 5))
#         self.set_count(command.get('-c', 100))
#         self.set_limit(command.get('-l'))
#         self.set_percent(command.get('-p'))
#
#     def command_management(self):
#         counter = 0
#         for video_path in self._video_paths:
#             if video_path is None:
#                 return
#             get_log_file(True, video_path, self.output_path[counter], self._format,
#                          self._rate, self._count, self._limit, self._percent)
#             counter = counter+1
#         print("Frames extraction completed successfully.")
#
#
#
# def cli_engine():
#     # Initialize colorama
#     init(autoreset=True)
#     big_title = pyfiglet.figlet_format("Video 2 Images")
#     print(Fore.BLUE + big_title)
#     print(welcome_message)
#     try:
#         while True:
#             command_input = input(cli_message)
#             if command_input in exit_flags:
#                 print("Bye Bye...")
#                 sys.exit(0)
#             user_input = UserInput(command_input)
#             if user_input.help:
#                 continue
#     except EOFError as e:
#         print("\nEnd of input received. Exiting.")
#
#
#
