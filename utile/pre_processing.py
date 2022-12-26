import cv2
import numpy as np
from typing import List


class PreProcessingPipeline:
    def __init__(
        self,
        method_to_gray: str = None,
        remove_annotation: bool = True,
        remove_line: str = "delete",
        normalization: str = "clahe",
        denoising: str = "fastNlMeans",
        gamma_correction: float = 2.2,
        background_uniformity: bool = False,
        sharpening: bool = False,
    ) -> None:
        """
        Constructor of the pre processing pipeline

        remove_annotation: If one wants to remove annotation in the pre-processing
        remove_line: Which method one wants to use to remove horizontal lines in the pre-processing (could be None)
        normalization: Which method one wants to use to normalize histograms in the pre-processing (could be None)
        denoising: Which method one wants to denoize images in the pre-processing (could be None)
        gamma_correction: Which value of gamma if one wants to gamma correct images in the pre-processing (could be None)
        background_uniformity: TODO
        sharpening: TODO

        returns: None
        """
        self.method_to_gray = method_to_gray
        self.remove_annotation = remove_annotation
        self.remove_line = remove_line
        self.normalization = normalization
        self.denoising = denoising
        self.gamma_correction = gamma_correction
        self.background_uniformity = background_uniformity
        self.sharpening = sharpening

    @staticmethod
    def _to_gray(image, method_to_gray: str = None) -> np.array:
        """
        Convert mammography sceening to gray

        image: The image to convert
        method_to_gray: Which method to use to convert to gray (None correponds to the open cv method
        TODO explore other ways to make it gray, e.g PCA cf. Impact of Image Enhancement Module for Analysis
        of Mammogram Images for Diagnostics of Breast Cancer)

        returns: The image in grayscale
        """
        if not method_to_gray:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def get_contours(image, thresh_low: int = 5, thresh_high: int = 255) -> List:
        """
        Get the list of contours of an image with opencv

        image: The screening mammography
        thresh_low: Lower bound for the threshold of the image
        thresh_high : Upper bound for the threshold of the image

        returns: The list of contours of the image
        """
        # Perform thresholding to create a binary image
        _, binary = cv2.threshold(image, thresh_low, thresh_high, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def draw_contours(contours: List, image: np.array, biggest: bool = True) -> tuple:
        """
        Draw contour on images

        contours: The list of the contour of the image
        image: The screening mammography
        biggest: If True, draws the biggest contour, else draw other contours

        returns: image with associate contours, binary mask of the contour
        """

        biggest_contour = max(contours, key=cv2.contourArea)
        # Create a mask image with the same size as the original image
        mask = np.zeros_like(image)
        # Convert the mask image to grayscale
        if len(image.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        if biggest:
            # Draw the biggest contour on the mask image
            # cv2.drawContours(mask_gray, [biggest_contour], -1, (255, 255, 255), -1)
            cv2.drawContours(mask_gray, [biggest_contour], -1, 255, -1)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(image, [biggest_contour], -1, (255, 0, 0), 3)

            # Set the pixels inside the removed contours to red
        # image[mask_gray == 255] = 255
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            for contour in contours:
                if cv2.contourArea(contour) != cv2.contourArea(biggest_contour):
                    cv2.drawContours(mask_gray, [contour], -1, (255, 255, 255), -1)
                    cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
        return image, mask_gray

    @staticmethod
    def get_horizontal_lines(
        image: np.array,
        LMIN: int = 400,
        LMAX: int = 500,
        minLineLength: int = 100,
        maxLineGap: int = 20,
        alpha: float = 0.1,
    ):
        """
        Draw and return horizontal lines if they exist (on some screenings). Segments are extended to lines

        image: The screening mammography
        LMIN: Minimum value in canny detection (cf. open cv canny documentation)
        LMAX: Maximum value in canny detection (cf. open cv canny documentation)
        minLineLength: The minimum length of lines to keep (cf. open cv HoughLinesP documentation)
        maxLineGap: The max gap between point of lines to keep (cf. open cv HoughLinesP documentation)
        alpha: Maximum horizontal orientation of lines to keep

        returns: image with associate lines, list of lines
        """

        edges = cv2.Canny(image, LMIN, LMAX)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            50,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap,
        )
        try:
            if not lines:
                return image, None
        except:
            pass
        image_line = image.copy()
        filtered_lines = []
        if len(lines) != 0:
            lines = sorted(
                lines,
                key=lambda x: np.sqrt(
                    (x[0][0] - x[0][2]) ** 2 + (x[0][1] - x[0][3]) ** 2
                ),
            )
            # Keep only the longest lines
            lines = lines[-min(6, len(lines)) :]

            # Iterate over the lines and draw them on the image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < alpha:
                    x2 = image.shape[1] - 1
                    x1 = 0
                    line = [[x1, y1, x2, y2]]
                    filtered_lines.append(line)
                    cv2.line(image_line, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if not filtered_lines:
                return image, None
        filtered_lines = sorted(filtered_lines, key=lambda x: x[0][1])
        return image_line, filtered_lines

    @staticmethod
    def split_images_lines(image: np.array, filtered_lines: List) -> List[np.array]:
        """
        Split image in list of images, delimited by the horizontal lines

        image: The screening mammography
        filtered_lines: Horizontal lines that are kept

        returns: List of splited images
        """
        # Initialize the list of images
        images = []
        filtered_lines = sorted(filtered_lines, key=lambda x: x[0][1])
        # Iterate over the lines and crop the image
        y1 = 0
        for line in filtered_lines:
            y2 = line[0][1]
            images.append(image[y1:y2, :])
            y1 = y2

        # Add the final image
        images.append(image[y1:, :])
        return images

    @staticmethod
    def restore_image(
        image: np.array,
        list_images: List[np.array],
        filtered_lines: List,
        method: str = "delete",
        thresh_low: float = 0.1,
        shape: int = 512,
    ):
        """
        Reconstruct image with horizontal white lines

        image: The screening mammography
        list_images: List of splited images according to horizontal lines
        filtered_lines: Lines to keep
        method: Method to restore the image (either delete rectangles of white bands, or opencv inpainting, TODO
        Masked ViT to restore parts of images)
        thresh_low: The part bands represent in the image to keep (to avoid deleting actual parts of images:
        TODO take average pixel value rather than the size of rectangle to identify white bands)
        shape: Length of the image

        returns: List of splited images
        """
        image_restored = image
        indexes = [
            i
            for i, elem in enumerate(list_images)
            if elem.shape[0] > int(thresh_low * shape)
        ]
        list_images = [elem[4:-4] for i, elem in enumerate(list_images) if i in indexes]
        if method == "delete":
            image_restored = cv2.resize(np.vstack(list_images), image.shape[:2])

        # TODO doesn't work for now (zones aren't correct)
        elif method == "opencv paint":

            zones_to_paint = [
                (filtered_lines[i - 1][0], filtered_lines[i][0])
                for i, _ in enumerate(list_images)
                if (i not in indexes) and (i - 1 >= 0)
            ]
            for zone in zones_to_paint:

                # Create a mask for the rectangle
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[zone[0][1] - 4 : zone[1][1] + 4, 0:512] = 255
                # Inpaint the image
                image_restored = cv2.inpaint(image_restored, mask, 3, cv2.INPAINT_TELEA)

        return image_restored

    @staticmethod
    def normalize(
        image: np.array,
        method: str = "clahe",
        clipLimit: float = 2.0,
        tileGridSize: tuple = (8, 8),
    ):
        """
        Normalize image's histogram

        image: The screening mammography
        method: Method used for normalization (global of clahe or TODO other methods)
        clipLimit: Clip limit for clahe
        tileGridSize: TODO explain

        returns: Normalized image
        """
        if method == "global":
            image = cv2.equalizeHist(image)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            image = clahe.apply(image)
        return image

    def _remove_annotation(self, image: np.array):
        """
        Remove annotation in image

        image: The screening mammography

        returns: Image without annotation
        """
        contours = self.get_contours(image)
        mask = self.draw_contours(contours, image, biggest=True)[1]
        return cv2.bitwise_and(image, mask)

    def pre_process(self):
        pass
