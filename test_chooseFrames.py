import os
import unittest
from unittest import TestCase

import numpy as np
from PIL import Image

from src.chooseFrames import save_frames_as_photos, create_mesh_by_ODM


def load_photos_from_directory(directory_path):
    photo_list = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add more extensions if needed

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return photo_list

    # Iterate through files in the directory
    for file_name in os.listdir(directory_path):
        if os.path.splitext(file_name)[1].lower() in valid_extensions:
            try:
                # Attempt to open the image file
                image_path = os.path.join(directory_path, file_name)
                image = Image.open(image_path)
                image_array = np.array(image)
                photo_list.append(image_array)
            except (IOError, OSError) as e:
                print(f"Error loading '{file_name}': {e}")

    return photo_list


class Test(TestCase):
    def setUp(self):
        self.load_path = "/Users/noyagendelman/Desktop/test/test_images"
        self.save_path = "/Users/noyagendelman/Desktop/test/test_images/test_save_frames"
        self.photos = load_photos_from_directory(self.load_path)
        self.photos_path_for_mesh = "/Users/noyagendelman/Desktop/test/iteration_5/v1"

    def test_save_frames_as_photos(self):
        save_frames_as_photos(self.photos, self.save_path)
        self.assertEqual(len(self.photos), len(self.photos), "Not all photos were saved correctly")

        # Check if saved files exist and are indeed images
        for i, photo in enumerate(self.photos):
            photo_path = os.path.join(self.save_path, f"frame_{i}.jpg")
            self.assertTrue(os.path.exists(photo_path),
                            f"frame {i} does not exist at {self.save_path}")

    def test_create_mesh_by_odm(self):
        create_mesh_by_ODM(self.photos_path_for_mesh)



class Test(TestCase):
    def test_create_mesh_by_odm(self):
        create_mesh_by_ODM("/Users/noyagendelman/Desktop/test/iteration_8/v1")


if __name__ == '__main__':
    unittest.main()



