# Exported from experiments/visualization_similarity/visualization_of_all_the three_techniques/visualization_twchniques_plus_atttention_using_mask.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import numpy as np
import cv2

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # Create a blank canvas (all black initially)
    canvas2 = np.zeros((height, width, 3), dtype=np.uint8)
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                canvas2[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return canvas2, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/val/val2/n01440764/ILSVRC2012_val_00003014.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Example binary patch mask (1 to keep, 0 to replace with black)
    patch_mask = np.random.choice([0, 1], size=(196,))  # Random binary mask for demonstration

    # Apply the patch mask
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using OpenCV
    cv2.imshow('Output Image with Black Patches', canvas)
    cv2.waitKey(0)  # Wait for key press to close the window
    cv2.destroyAllWindows()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 1
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # Create a blank canvas (all black initially)
    canvas2 = np.zeros((height, width, 3), dtype=np.uint8)
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                canvas2[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return canvas2, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/val/val2/n01773157/ILSVRC2012_val_00004118.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Example binary patch mask (1 to keep, 0 to replace with black)
    patch_mask = np.random.choice([0, 1], size=(196,))  # Random binary mask for demonstration

    # Apply the patch mask
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 2



# %% Original cell 3
import numpy as np

# Given array of True and False
bool_array = np.array([False, False, False,  True,  True,  True,  True,  True,  True,  True,
         False, False, False, False, False, False, False, False,  True,  True,
          True,  True,  True,  True, False, False, False, False, False, False,
         False,  True,  True,  True,  True,  True,  True,  True, False, False,
         False, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False,  True, False,
          True,  True,  True,  True,  True,  True,  True,  True,  True, False,
         False, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True, False,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True, False, False,  True,  True,  True,
          True,  True,  True,  True,  True, False, False, False, False, False,
         False,  True,  True,  True,  True,  True,  True,  True, False, False,
         False, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True])

# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 4
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/val/val4/valllllll/ILSVRC2012_val_00002064.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 5



# %% Original cell 6
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.85   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/val/val4/valllllll/ILSVRC2012_val_00002064.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
         1, 0, 1, 0]
    
    
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 7
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    """

    patch_size = 16
    height, width = features_img1.shape

    # Ensure patch_mask size is correct
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"

    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # Create white canvas
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Blend original image with white canvas
    alpha = 0.2
    beta = 0.85
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)

    count2 = 0
    patch_idx = 0

    for i in range(num_patches_y):
        for j in range(num_patches_x):

            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            if patch_mask[patch_idx] == 1:
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1

            patch_idx += 1

    return blended_image, count2


def main():

    # image_path = '/home/viraj/my_project/val/val4/valllllll/ILSVRC2012_val_00002064.JPEG'
    image_path= "/home/viraj/my_project/val/val4/valllllll/ILSVRC2012_val_00000570.JPEG"
    img = cv2.imread(image_path)

    # Resize to 224x224
    img = cv2.resize(img, (224, 224))

    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Your 196-length patch mask
    patch_mask = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
         1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
         1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,
         0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1,
         1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1]

    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # -------------------------------
    # ADD BLACK BORDER HERE
    # -------------------------------
    border_size = 1  # Thickness of border

    canvas_with_border = cv2.copyMakeBorder(
        canvas,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black color (BGR)
    )

    # Save output
    output_image_path = 'output_image_with_black_border.jpeg'
    cv2.imwrite(output_image_path, canvas_with_border)

    # Display
    plt.imshow(cv2.cvtColor(canvas_with_border, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")


if __name__ == "__main__":
    main()


# %% Original cell 8



# %% Original cell 9



# %% Original cell 10



# %% Original cell 11
import numpy as np

# Given array of True and False
bool_array = np.array([ True,  True,  True,  True, False,  True,  True,  True,  True,  True,
         False, False,  True,  True,  True,  True, False, False, False,  True,
          True,  True,  True, False,  True,  True,  True, False,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True, False,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True, False,  True, False,  True,  True,
          True,  True,  True,  True,  True, False,  True,  True,  True, False,
         False,  True,  True,  True,  True,  True,  True, False,  True, False,
          True, False, False, False,  True,  True,  True,  True,  True, False,
          True,  True, False, False, False, False,  True,  True,  True,  True,
          True,  True, False, False, False, False,  True, False, False,  True,
          True,  True,  True,  True,  True,  True,  True, False,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False,  True,
          True,  True,  True,  True,  True,  True,  True])

# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 12
import numpy as np

def create_new_array(array1, array2):
    """
    Create a new array based on the conditions:
    - If the value in array1 is 1, check the corresponding position in array2.
    - If the value in array2 at that position is 1, place 1 in the new array.
    - Otherwise, place 0 in the new array.

    Parameters:
    array1: numpy array of shape (196,)
    array2: numpy array of dynamic length

    Returns:
    new_array: numpy array with values based on the conditions
    """
    # Ensure array2 is a NumPy array for consistent operations

    # Initialize the new array with the same length as array1
    empty_list = []
    j=0

    # Iterate through array1 and create new_array based on conditions
    for i in range(len(array1)):
        if array1[i] == 0:
            empty_list.append(0)
        else:
            if array1[i]==1 and array2[j]==0:
                empty_list.append(0)
                j+=1
            else:
                empty_list.append(1)
                j+=1
                

    return empty_list

# Example usage
array1 = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1]


array2 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

new_array = create_new_array(array1, array2)

print("New Array:", new_array)


# %% Original cell 13
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path =  '/home/viraj/my_project/val/val4/valllllll/ILSVRC2012_val_00002064.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

 
    
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 14
import numpy as np


# Given array of True and False
bool_array = np.array([ True,  True,  True,  True, False,  True,  True,  True,  True,  True,
         False, False,  True,  True,  True,  True, False, False, False,  True,
          True,  True,  True, False,  True,  True,  True, False,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True, False,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True, False,  True, False,  True,  True,
          True,  True,  True,  True,  True, False,  True,  True,  True, False,
         False,  True,  True,  True,  True,  True,  True, False,  True, False,
          True, False, False, False,  True,  True,  True,  True,  True, False,
          True,  True, False, False, False, False,  True,  True,  True,  True,
          True,  True, False, False, False, False,  True, False, False,  True,
          True,  True,  True,  True,  True,  True,  True, False,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False,  True,
          True,  True,  True,  True,  True,  True,  True])


# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 15
import numpy as np

def create_new_array(array1, array2):
    """
    Create a new array based on the conditions:
    - If the value in array1 is 1, check the corresponding position in array2.
    - If the value in array2 at that position is 1, place 1 in the new array.
    - Otherwise, place 0 in the new array.

    Parameters:
    array1: numpy array of shape (196,)
    array2: numpy array of dynamic length

    Returns:
    new_array: numpy array with values based on the conditions
    """
    # Ensure array2 is a NumPy array for consistent operations

    # Initialize the new array with the same length as array1
    empty_list = []
    j=0

    # Iterate through array1 and create new_array based on conditions
    for i in range(len(array1)):
        if array1[i] == 0:
            empty_list.append(0)
        else:
            if array1[i]==1 and array2[j]==0:
                empty_list.append(0)
                j+=1
            else:
                empty_list.append(1)
                j+=1
                

    return empty_list

# Example usage
array1 = [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

array2 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

new_array = create_new_array(array1, array2)

print("New Array:", new_array)


# %% Original cell 16



# %% Original cell 17
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/val/val4/valllllll/ILSVRC2012_val_00002064.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask =  [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 18
import numpy as np

# Given array of True and False
bool_array = np.array([ True,  True,  True, False,  True,  True,  True,  True,  True,  True,
          True,  True,  True, False,  True,  True,  True, False,  True,  True,
          True,  True,  True, False,  True,  True, False,  True,  True,  True,
         False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False,  True,  True,  True,  True,  True,  True,  True, False,
          True,  True,  True, False,  True, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True, False,  True,  True, False,  True,
          True,  True, False,  True,  True,  True, False,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True, False,
         False,  True,  True, False,  True,  True,  True,  True,  True, False,
          True,  True,  True,  True, False, False, False,  True,  True,  True,
          True, False,  True,  True,  True,  True, False,  True,  True,  True,
          True, False, False,  True,  True])

# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 19
import numpy as np

def create_new_array(array1, array2):
    """
    Create a new array based on the conditions:
    - If the value in array1 is 1, check the corresponding position in array2.
    - If the value in array2 at that position is 1, place 1 in the new array.
    - Otherwise, place 0 in the new array.

    Parameters:
    array1: numpy array of shape (196,)
    array2: numpy array of dynamic length

    Returns:
    new_array: numpy array with values based on the conditions
    """
    # Ensure array2 is a NumPy array for consistent operations

    # Initialize the new array with the same length as array1
    empty_list = []
    j=0

    # Iterate through array1 and create new_array based on conditions
    for i in range(len(array1)):
        if array1[i] == 0:
            empty_list.append(0)
        else:
            if array1[i]==1 and array2[j]==0:
                empty_list.append(0)
                j+=1
            else:
                empty_list.append(1)
                j+=1
                

    return empty_list

# Example usage
array1 =  [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
array2 =[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1]

new_array = create_new_array(array1, array2)

print("New Array:", new_array)


# %% Original cell 20
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/val/val4/valllllll/ILSVRC2012_val_00002064.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 21
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with white patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace with white (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with white.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # Create a blank canvas filled with almost white color
    canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245  # Almost white background
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with white
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                canvas2[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return canvas2, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/dog_image/image_1/ILSVRC2012_test_00089861.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_white_shade.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title('Output Image with White Shade on Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 22
import os
import cv2
import matplotlib.pyplot as plt

def visualize_images_from_folders(root_folder):
    """
    Traverse the root folder to visualize images inside subfolders.

    Parameters:
    root_folder (str): The root directory containing subfolders with images.
    """
    
    # Iterate through all subdirectories and files in the root folder
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            # Check if the file is an image by extension
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # Get the full file path
                file_path = os.path.join(subdir, file)
                print(f"Visualizing: {file_path}")
                
                # Read the image using OpenCV
                img = cv2.imread(file_path)
                
                # Convert BGR to RGB for proper visualization in matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Display the image using matplotlib
                plt.figure()
                plt.imshow(img_rgb)
                plt.title(f"Image from: {file_path}")
                plt.axis('off')  # Turn off axis
                plt.show()

if __name__ == "__main__":
    # Path to the root folder that contains subfolders with images
    root_folder = '/home/viraj/my_project/val/val2/'
    
    # Visualize images
    visualize_images_from_folders(root_folder)


# %% Original cell 23
import numpy as np

# Given array of True and False
bool_array = np.array([ True,  True,  True,  True,  True,  True, False,  True, False, False,
         False, False, False,  True,  True,  True, False,  True,  True,  True,
          True, False, False,  True,  True, False,  True, False,  True,  True,
          True, False, False,  True,  True,  True, False, False, False, False,
         False, False,  True,  True, False,  True,  True,  True, False, False,
          True,  True,  True,  True, False, False,  True,  True, False,  True,
         False,  True, False, False,  True,  True,  True,  True, False,  True,
          True, False,  True,  True, False, False, False,  True,  True,  True,
          True,  True, False, False, False, False, False,  True, False, False,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True, False, False,  True,  True,  True,  True, False, False,
         False, False,  True, False,  True,  True, False, False, False,  True,
         False, False,  True,  True, False, False, False, False,  True,  True,
          True,  True, False, False, False,  True,  True, False, False, False,
         False, False,  True,  True,  True,  True, False, False, False, False,
          True,  True,  True, False, False, False, False, False,  True,  True,
         False, False, False, False, False,  True, False, False, False, False,
         False, False, False, False, False, False, False,  True, False, False,
         False, False,  True,  True, False,  True, False,  True,  True, False,
          True,  True,  True, False, False,  True])

# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 24
import numpy as np

def create_new_array(array1, array2):
    """
    Create a new array based on the conditions:
    - If the value in array1 is 1, check the corresponding position in array2.
    - If the value in array2 at that position is 1, place 1 in the new array.
    - Otherwise, place 0 in the new array.

    Parameters:
    array1: numpy array of shape (196,)
    array2: numpy array of dynamic length

    Returns:
    new_array: numpy array with values based on the conditions
    """
    # Ensure array2 is a NumPy array for consistent operations

    # Initialize the new array with the same length as array1
    empty_list = []
    j=0

    # Iterate through array1 and create new_array based on conditions
    for i in range(len(array1)):
        if array1[i] == 0:
            empty_list.append(0)
        else:
            if array1[i]==1 and array2[j]==0:
                empty_list.append(0)
                j+=1
            else:
                empty_list.append(1)
                j+=1
                

    return empty_list

# Example usage
array1 =  [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
array2 =[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]

new_array = create_new_array(array1, array2)

print("New Array:", new_array)


# %% Original cell 25
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/val/val5/vallll/ILSVRC2012_val_00006306.JPEG'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 26
import numpy as np

# Given array of True and False
bool_array = np.array([True,  True,  True,  True,  True,  True,  True,  True, False,
          True,  True,  True,  True,  True,  True, False, False, False, False,
          True,  True,  True,  True,  True,  True,  True,  True, False,  True,
         False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True, False,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True, False,  True,  True,  True,  True,  True,  True,  True, False,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False, False,  True,  True,  True,  True, False, False, False,
         False,  True,  True,  True,  True, False, False, False, False,  True,
          True,  True,  True, False, False, False, False, False, False, False,
          True,  True,  True,  True,  True, False, False, False, False, False,
         False, False, False,  True,  True,  True,  True,  True,  True,  True,
          True, False,  True,  True,  True,  True,  True,  True])

# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 27
import numpy as np

def create_new_array(array1, array2):
    """
    Create a new array based on the conditions:
    - If the value in array1 is 1, check the corresponding position in array2.
    - If the value in array2 at that position is 1, place 1 in the new array.
    - Otherwise, place 0 in the new array.

    Parameters:
    array1: numpy array of shape (196,)
    array2: numpy array of dynamic length

    Returns:
    new_array: numpy array with values based on the conditions
    """
    # Ensure array2 is a NumPy array for consistent operations

    # Initialize the new array with the same length as array1
    empty_list = []
    j=0

    # Iterate through array1 and create new_array based on conditions
    for i in range(len(array1)):
        if array1[i] == 0:
            empty_list.append(0)
        else:
            if array1[i]==1 and array2[j]==0:
                empty_list.append(0)
                j+=1
            else:
                empty_list.append(1)
                j+=1
                

    return empty_list

# Example usage
array1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
array2 =[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]

new_array = create_new_array(array1, array2)

print("New Array:", new_array)


# %% Original cell 28
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/dog_image/image_1/ILSVRC2012_val_00008284.jpg'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 29
import numpy as np

# Given array of True and False
bool_array = np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True, False, False,  True,  True,  True,  True,  True, False,
          True, False,  True,  True,  True,  True,  True,  True,  True,  True,
         False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False,  True,  True,  True,  True,  True,  True, False,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True, False, False,  True, False,  True,  True, False,
          True,  True,  True,  True,  True,  True,  True, False, False, False,
          True,  True, False,  True, False,  True, False, False,  True, False,
         False, False, False, False, False, False, False, False,  True,  True,
         False])

# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 30
import numpy as np

def create_new_array(array1, array2):
    """
    Create a new array based on the conditions:
    - If the value in array1 is 1, check the corresponding position in array2.
    - If the value in array2 at that position is 1, place 1 in the new array.
    - Otherwise, place 0 in the new array.

    Parameters:
    array1: numpy array of shape (196,)
    array2: numpy array of dynamic length

    Returns:
    new_array: numpy array with values based on the conditions
    """
    # Ensure array2 is a NumPy array for consistent operations

    # Initialize the new array with the same length as array1
    empty_list = []
    j=0

    # Iterate through array1 and create new_array based on conditions
    for i in range(len(array1)):
        if array1[i] == 0:
            empty_list.append(0)
        else:
            if array1[i]==1 and array2[j]==0:
                empty_list.append(0)
                j+=1
            else:
                empty_list.append(1)
                j+=1
                

    return empty_list

# Example usage
array1 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
array2 =[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]

new_array = create_new_array(array1, array2)

print("New Array:", new_array)


# %% Original cell 31
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/dog_image/image_1/ILSVRC2012_val_00008284.jpg'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask =  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 32
import numpy as np

# Given array of True and False
bool_array = np.array([ True, False, False, False, False,  True,  True, False, False, False,
          True, False,  True,  True,  True, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False,  True,  True,
          True,  True,  True,  True,  True, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False, False, False,
          True,  True,  True,  True,  True,  True,  True,  True,  True, False,
          True,  True,  True,  True,  True,  True,  True, False, False, False,
         False, False, False])

# Convert to 0 and 1 using astype
binary_array = bool_array.astype(int)

# Convert the binary array to a list of strings
binary_list = binary_array.astype(str)

# Join the list with commas
binary_string = ', '.join(binary_list)

# Print the result
print(binary_string)


# %% Original cell 33
import numpy as np

def create_new_array(array1, array2):
    """
    Create a new array based on the conditions:
    - If the value in array1 is 1, check the corresponding position in array2.
    - If the value in array2 at that position is 1, place 1 in the new array.
    - Otherwise, place 0 in the new array.

    Parameters:
    array1: numpy array of shape (196,)
    array2: numpy array of dynamic length

    Returns:
    new_array: numpy array with values based on the conditions
    """
    # Ensure array2 is a NumPy array for consistent operations

    # Initialize the new array with the same length as array1
    empty_list = []
    j=0

    # Iterate through array1 and create new_array based on conditions
    for i in range(len(array1)):
        if array1[i] == 0:
            empty_list.append(0)
        else:
            if array1[i]==1 and array2[j]==0:
                empty_list.append(0)
                j+=1
            else:
                empty_list.append(1)
                j+=1
                

    return empty_list

# Example usage
array1 =  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
array2 =[ 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

new_array = create_new_array(array1, array2)

print("New Array:", new_array)


# %% Original cell 34
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/dog_image/image_1/ILSVRC2012_val_00008284.jpg'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()


# %% Original cell 35
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canva(features_img1, img1, patch_mask):
    """
    Applies a mask to the image, replacing certain patches with black patches based on the patch_mask.
    
    Parameters:
    features_img1: NumPy array (H, W) - Grayscale feature image.
    img1: NumPy array (H, W, 3) - Original RGB image.
    patch_mask: 1D binary array (196,) - Binary mask to indicate which patches to keep (1) or replace (0).
    
    Returns:
    canvas2: NumPy array (H, W, 3) - Processed image with selected patches kept and others replaced with black.
    count2: int - Number of patches retained.
    """
    
    # Parameters
    patch_size = 16
    height, width = features_img1.shape
    
    # Ensure patch_mask is correct size
    assert len(patch_mask) == (height // patch_size) * (width // patch_size), "Patch mask size mismatch"
    
    # Calculate the number of patches along each dimension
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size

    # # Create a blank canvas (all black initially)
    # canvas2=img1
    # canvas2 = np.ones((height, width, 3), dtype=np.uint8) * 245
    light_white_canvas = np.ones((height, width, 3), dtype=np.uint8)
    light_white_canvas[:, :] = [255, 255, 255]  # RGB for light pink
    
    # Blend the original image with the light white canvas
    # Use cv2.addWeighted to blend the two images
    alpha = 0.2  # The weight of the original image
    beta = 0.8   # The weight of the white shade
    blended_image = cv2.addWeighted(img1, alpha, light_white_canvas, beta, 0)


    
    count2 = 0
    patch_idx = 0

    # Process each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the coordinates of the current patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            # Check if the current patch should be kept or replaced with black
            if patch_mask[patch_idx] == 1:
                # Keep the patch from the original image
                patch = img1[y_start:y_end, x_start:x_end]
                blended_image[y_start:y_end, x_start:x_end] = patch
                count2 += 1  # Count the patches that are retained

            # Increment the patch index
            patch_idx += 1

    return blended_image, count2

# Example usage
def main():
    # Load the image using OpenCV
    image_path = '/home/viraj/my_project/dog_image/image_1/ILSVRC2012_val_00008284.jpg'
    img = cv2.imread(image_path)
    
    # Resize image to 224x224 if it's not already in the right size
    img = cv2.resize(img, (224, 224))
    
    # Convert to grayscale for feature image
    features_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use your provided binary patch mask
    patch_mask = np.random.randint(1, 2, size=196)
    # patch_mask = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    canvas, kept_patch_count = canva(features_img1, img, patch_mask)

    # Save the resulting image using OpenCV
    output_image_path = 'output_image_with_black_patches.jpeg'
    cv2.imwrite(output_image_path, canvas)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title('Output Image with Black Patches')
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of patches kept: {kept_patch_count}")

if __name__ == "__main__":
    main()
