import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import correlate
from scipy.signal import correlate2d


def normalize_matrix(image: np.ndarray, upperbound: int = 255, lowerbound: int = 0, gray_value: int = 0) -> np.ndarray:
    """This function normalizes our matrix between two custom values
    right way to iterate on a numpy array and change some values
    extracted from https://numpy.org/devdocs/reference/arrays.nditer.html
    values bigger than 255 would change to 255 and less than 0 would change to 0
    Args:
        image(np.ndarray): An image as numpy array
        upperbound(int): upper bound of filtering
        lowerbound(int): lower bound of filtering
        gray_value(int): optional value to set for between upper and lower bound
    Returns:
        np.ndarray : image after normalization

    """
    with np.nditer(image, op_flags=['readwrite']) as it:
        for i in it:
            if i > upperbound:
                i[...] = 255
            elif i < lowerbound:
                i[...] = 0
            elif gray_value != 0:
                i[...] = gray_value
    return image


def laplace_filter(image: np.ndarray) -> np.ndarray:
    """Implementation of Laplacian filter
    Args:
        image(np.ndarray): image as numpy array

    Returns:
        np.ndarray :  image after filtering
    """
    laplace_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    after_laplace_filter = correlate2d(image, laplace_kernel, mode='same', boundary='fill')

    return after_laplace_filter


def image_sharpening(image: np.ndarray) -> np.ndarray:
    """Simple function to sharpen a given image using a laplace 3x3 kernel and normalization function

    Args:
        image(np.ndarray): an image as np.ndarray read by cv2 or PILLOW

    Returns:
        np.ndarray: sharpened image using laplace method

    """
    sharpened_image = normalize_matrix(laplace_filter(image))

    # We have to do this transformation in order to use cv2.imshow() if you use matplotlib there is no need
    # for this line
    # sharpened_image = np.array(sharpened_image, dtype=np.uint8)
    return sharpened_image


def sobel_operator(image: np.ndarray) -> np.ndarray:
    """Simple implementation of Sobel edge detection

    Args:
        image(np.ndarray): our input image as numpy array

    Returns:
        np.ndarray: Image after Sobel edge detection
    """
    correlated_image = list()
    sobel_kernels = [np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])]

    # Calculating Dx and Dy
    for kernel in sobel_kernels:
        # Using filters.correlate instead of correlate2d to avoid 0 paddings and use automatic normalization
        correlated_image.append(correlate(image, kernel))

    correlated_image = list(map(np.abs, correlated_image))
    final_image = normalize_matrix(np.add(correlated_image[0], correlated_image[1]))
    return final_image


def pixel_theta(image: np.ndarray) -> np.ndarray:
    """Calculation of theta in the Sobel filter formula , using arctan()
    Args:
        image(np.ndarray): an image as input

    Returns:
        np.ndarray : a numpy array filled with thetas for each pixel
    """

    correlated_image = list()
    sobel_kernel = [np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])]

    for kernel in sobel_kernel:
        correlated_image.append(correlate2d(image, kernel, mode='same', boundary='fill'))

    # in order to avoid runtime warning
    np.seterr(divide='ignore', invalid='ignore')
    thetas_for_each_pixel = np.array(list(map(np.arctan, np.divide(correlated_image[1], correlated_image[0]))))
    return thetas_for_each_pixel


def blur_function(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """This function will return the blurry version of the original image using average function

    Args:
        image(np.ndarray): our image as numpy array
        kernel_size(int): size of our kernel , n for nxn kernel

    Returns:
        np.ndarray : blurry image
    """

    custom_kernel = np.ones((kernel_size, kernel_size), dtype=np.float) / kernel_size ** 2
    blurry_image = correlate2d(image, custom_kernel, mode='same', boundary='fill')
    return blurry_image


def unsharp_mask(image: np.ndarray, blur_kernel_size: int = 4) -> np.ndarray:
    """Simple function to apply unsharp masking filter on an image

    Args:
        image(np.ndarray): input image as numpy array
        blur_kernel_size(int): this would be the kernel_size in blur_function(image, kernel_size)

    Returns:
        np.ndarray: our image after unsharp masking
    """
    mask = np.subtract(image, blur_function(image, blur_kernel_size))
    unsharpened_image = normalize_matrix(image + mask)
    return unsharpened_image


def high_boost_filtering(image: np.ndarray, k: int, blur_kernel_size: int = 4) -> np.ndarray:
    """High boost filtering on input image with custom kernel size and blur kernel size

    Args:
        image(np.ndarray): our input image as numpy array
        k(int): k in the formula : image + k *  mask
        blur_kernel_size(int): blur function kernel size

    Returns:
        np.ndarray: image after the high boost filtering
    """
    high_boosted_image = image + k * np.subtract(image, blur_function(image, blur_kernel_size))
    high_boosted_image = normalize_matrix(high_boosted_image)

    return high_boosted_image


def gaussian_blur_filter(image: np.ndarray) -> np.ndarray:
    """Gausian blur function using pre-made kernel instead of original formula
    Args:
        image(np.ndarray): input image as numpy array

    Returns:
        np.ndarray: blurry image
    """
    gaussian_kernel = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]], dtype=np.float32) / 273
    blurry_image = correlate(image, gaussian_kernel)

    return blurry_image


def map_to_eight_thetas(thetas_matrix: np.ndarray) -> np.ndarray:
    """Mapping our thetas to 8 degrees
    Args:
        thetas_matrix(np.ndarray):  array of calculated thetas

    Returns:
        np.ndarray: mapped thetas
    """
    # Saving pi value
    pi = np.pi
    # First we Mirror one side of the circle on the other side just to simplify the problem
    with np.nditer(thetas_matrix, op_flags=['readwrite']) as it:
        for i in it:
            if i < 0:
                i[...] += pi
    # Mapping process
    with np.nditer(thetas_matrix, op_flags=['readwrite']) as it:
        for i in it:
            if (0 <= i < pi / 8) or (7 * pi / 8 < i <= pi):
                i[...] = pi / -2
            elif pi / 8 < i <= 3 * pi / 8:
                i[...] = pi / -4
            elif 3 * pi / 8 < i <= 5 * pi / 8:
                i[...] = 0
            elif 5 * pi / 8 < i <= 7 * pi / 8:
                i[...] = pi / 4
            else:
                i[...] = 0
    return thetas_matrix


def non_maximum_suppresion(mapped_thetas: np.ndarray, image: np.ndarray) -> np.ndarray:
    """ Non maximum suppresion operation on mapped thetas
    Args:
        mapped_thetas(np.ndarray): mapped to 8 degrees thetas
        image(np.ndarray): original image

    Returns:
        np.ndarray: new image after operation
    """
    new_image = np.zeros(image.shape)
    neighbor_1 = 0
    neighbor_2 = 255

    with np.nditer(image, op_flags=['readwrite'], flags=['multi_index']) as it:
        for i in it:
            try:
                x, y = it.multi_index
                if mapped_thetas[x, y] == (np.pi / -2):
                    neighbor_2 = image[x, y + 1]
                    neighbor_1 = image[x, y - 1]
                elif mapped_thetas[x, y] == (np.pi / -4):
                    neighbor_2 = image[x + 1, y + 1]
                    neighbor_1 = image[x - 1, y - 1]
                elif mapped_thetas[x, y] == 0:
                    neighbor_2 = image[x + 1, y]
                    neighbor_1 = image[x - 1, y]
                elif mapped_thetas[x, y] == (np.pi / 4):
                    neighbor_2 = image[x - 1, y + 1]
                    neighbor_1 = image[x + 1, y - 1]
                if (i > neighbor_1) and (i > neighbor_2):
                    new_image[x, y] = i
                else:
                    pass
            except IndexError:
                pass
    return new_image


def double_threshold(image: np.ndarray, higher_ratio: float, lower_ratio: float) -> np.ndarray:
    """A double thereshould function between two rates

    Args:
        image(np.ndarray): image as numpy array
        higher_ratio(float): Higher ratio to find upper bound
        lower_ratio(float): Lower ratio to find lower bound

    Returns:
        np.ndarray: image after threshold appliance
    """

    upper_bound, lower_bound = image.max() * higher_ratio, image.max() * higher_ratio * lower_ratio
    after_threshold = normalize_matrix(image, upper_bound, lower_bound, 100)

    return after_threshold


def leftover_threshold(image: np.ndarray) -> np.ndarray:
    """Leftover thresholding operation function
    Args:
        image(np.ndarray): input image as numpy array

    Returns:
        np.ndarray: image after operation
    """
    with np.nditer(image, op_flags=['readwrite'], flags=['multi_index']) as it:
        for i in it:
            try:
                x, y = it.multi_index
                if 0 < i < 255:
                    sum_of_logical_bounds = ((image[x + 1, y - 1] == 255) or (image[x + 1, y] == 255) or
                                             (image[x + 1, y + 1] == 255) or (image[x, y - 1] == 255) or (
                                                         image[x, y + 1] == 255) or
                                             (image[x - 1, y - 1] == 255) or (image[x - 1, y] == 255) or (
                                                         image[x - 1, y + 1] == 255))
                    if sum_of_logical_bounds:
                        i[...] = 255
                    else:
                        i[...] = 0
            except IndexError:
                pass
    return image


def custom_canny(image: np.ndarray, higherratio: float, lowerratio: float) -> np.ndarray:
    """Implementation of a custom Canny edge detection
    Args:
        image(np.ndarray): input image
        higherratio: ratio for double threshold
        lowerratio: ratio for double threshold

    Returns:
        np.ndarray: edge detected image
    """
    # Wrapping up our previous functions
    image_ = gaussian_blur_filter(image)
    mapped_thetas = map_to_eight_thetas(pixel_theta(image_))
    sobel_blurry_image = sobel_operator(image_)
    sobel_blurry_image = gaussian_blur_filter(sobel_blurry_image)
    canny_result = non_maximum_suppresion(mapped_thetas, sobel_blurry_image)
    canny_result = leftover_threshold(double_threshold(canny_result, higherratio, lowerratio))

    return canny_result


def power_transform(image: np.ndarray, gamma_: float) -> np.ndarray:
    """Gamma correction function
    Args:
        image(np.ndarray):input image
        gamma_(float): Gamma correction ratio

    Returns:
        np.ndarray: image after transformation
    """
    gamma_corrected = np.array(255 * (image / 255) ** gamma_, dtype='uint8')
    return gamma_corrected


def bone_scan_enhancement(image: np.ndarray) -> np.ndarray:
    """ Funtion to enhance the quality of bone scans
    Args:
        image(np.ndarray): input image

    Returns:
        np.ndarray:  enhanced image
    """
    sharpened_image = image_sharpening(image)
    sobel_blur = blur_function(sobel_operator(image), 5)

    # Creating the mask
    enhanced_image = np.multiply(sharpened_image, sobel_blur, dtype=float)
    # Scale between 0 and 255
    enhanced_image /= enhanced_image.max() / 255.0
    # Sum of image and mask
    enhanced_image += image
    # Removing values greater and 255 and lower than 0
    enhanced_image = normalize_matrix(enhanced_image)
    # Power law Transformation
    enhanced_image = power_transform(enhanced_image, 0.9)
    return enhanced_image


def plot(image, name):
    plt.title(name)
    plt.imshow(image, cmap='gray')
    plt.show()


# This block is only generated for testing purposes
if __name__ == '__main__':
    skeleton = cv2.imread('sample_images/skeleton.tif', 0).astype(np.float32)
    camera_man = cv2.imread('sample_images/cameraman.tif', 0).astype(np.float32)

    plot(camera_man, "original cameraman")
    plot(image_sharpening(camera_man), "Image Sharpening")
    plot(sobel_operator(camera_man), "Sobel Edge Detection")
    plot(pixel_theta(camera_man), "Theta of Sobel formula")
    plot(unsharp_mask(camera_man), "Unsharp Masking")
    plot(high_boost_filtering(camera_man, 3), "High Boost Filtering")
    plot(skeleton, "original medical photo")
    plot(bone_scan_enhancement(skeleton), "Bone Scan Enhancement")
    plot(custom_canny(camera_man, 0.088, 0.05), "Canny Edge Detection")

