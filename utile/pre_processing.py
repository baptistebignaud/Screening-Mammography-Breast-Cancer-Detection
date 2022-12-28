import cv2
import numpy as np
from typing import List


class PreProcessingPipeline:
    def __init__(
        self,
        method_to_gray: str = "default",
        remove_annotation: bool = True,
        remove_line: bool = False,
        normalization: bool = True,
        denoising: bool = False,
        sharpening: bool = False,
        gamma_correction: bool = False,
        inverted_image: bool = False,
        **methods_args,
    ) -> None:
        """
        Constructor of the pre processing pipeline

        method_to_gray: Which method to use to convert image in gray scale
        remove_annotation: If one wants to remove annotation in the pre-processing
        remove_line: If one wants to use to remove horizontal white bands in the pre-processing
        normalization: If one wants to use to normalize histograms in the pre-processing
        denoising: If one wants to to denoize images in the pre-processing
        gamma_correction: If one wants to gamma correct images in the pre-processing
        sharpening: TODO
        **methods_args: All possible parameters for functions in pre-processing

        returns: None
        """
        self.method_to_gray = method_to_gray
        self.remove_annotation = remove_annotation
        self.remove_line = remove_line
        self.normalization = normalization
        self.denoising = denoising
        self.sharpening = sharpening
        self.gamma_correction = gamma_correction
        self.inverted_image = inverted_image

        # Default values for preprocessing that you can change in calling the constructor of the Pipeline class
        # Methods to adopt in the pipeline for each step
        self.denoising_method = "NlMD"
        self.normalization_method = "clahe"
        self.remove_line_method = "delete"

        # Arguments about contours detection
        self.contours_low = 5
        self.contours_high = 255

        # Arguments about gamma correction
        self.gamma_correction_value = 2.2

        # Arguments about detection of lines
        self.horizontal_lines_LMIN = 400
        self.horizontal_lines_LMAX = 500
        self.horizontal_lines_minLineLength = 100
        self.horizontal_lines_maxLineGap = 20
        self.horizontal_lines_alpha = 0.1
        self.houghlineP_threshold = 50

        # Arguments for normalization with CLAE
        self.clahe_clipLimit = 2.0
        self.clahe_tileGridSize = (10, 10)

        # Arguments for denoising
        self.denoise_h = 3
        self.denoise_block_size = 7
        self.denoise_search_window = 21

        # Possibilty to adjust parameters
        self.__dict__.update(methods_args)

    def pre_process(
        self, images: np.array or List[np.array]
    ) -> List[np.array] or np.array:
        """
        Global preprocessing function

        images: Image or list of images to pre-process

        returns: Image or list of images pre-processed
        """
        if not isinstance(images, List):
            images = [images]

        # Apply pre-processing for each image
        for i, image in enumerate(images):
            image = self._to_gray(image, method_to_gray=self.method_to_gray)

            if self.remove_annotation:
                image, mask = self._remove_annotation(
                    image, thresh_high=self.contours_high, thresh_low=self.contours_low
                )

            if self.remove_line:
                image = self._remove_line(
                    image,
                    method=self.remove_line_method,
                    LMIN=self.horizontal_lines_LMIN,
                    LMAX=self.horizontal_lines_LMAX,
                    minLineLength=self.horizontal_lines_minLineLength,
                    maxLineGap=self.horizontal_lines_maxLineGap,
                    alpha=self.horizontal_lines_alpha,
                )

            if self.normalization:
                if self.normalization_method == "clahe":
                    image = self.normalize(
                        image,
                        method=self.normalization_method,
                        clahe_clipLimit=self.clahe_clipLimit,
                        clahe_tileGridSize=self.clahe_tileGridSize,
                    )
                elif self.normalization_method == "global":
                    image = self.normalize(image, method=self.normalization_method)

            if self.denoising:
                image = self.denoise(
                    image,
                    method=self.denoising_method,
                    h=self.denoise_h,
                    block_size=self.denoise_block_size,
                    search_window=self.denoise_search_window,
                )

            if self.gamma_correction:
                image = self.gamma_correct(image, gamma=self.gamma_correction_value)

            if self.sharpening:
                # TODO #image = self.
                pass

            image = cv2.bitwise_and(image, mask)

            if self.inverted_image:
                image = self.invert_image(image)

            # Change image by pre processed image
            images[i] = image

        if len(images) == 1:
            return images[0]
        return images

    def _remove_annotation(
        self, image: np.array, thresh_high: int = 255, thresh_low: int = 5
    ):
        """
        Remove annotation in image

        image: The screening mammography

        returns: Image without annotation
        """
        contours = self.get_contours(
            image, thresh_low=thresh_low, thresh_high=thresh_high
        )
        mask = self.draw_contours(contours, image, biggest=True)[1]
        return cv2.bitwise_and(image, mask), mask

    def _remove_line(
        self,
        image: np.array,
        LMIN,
        LMAX,
        minLineLength,
        maxLineGap,
        alpha,
        method: str = "delete",
    ):
        """
        Very experimetal, remove horizontal white bands in some images

        image: The screening mammography
        method: The method to remove white bands

        returns: Image with horizontal white bands removed
        """
        # Get horizontal lines in mammography
        _, lines = self.get_horizontal_lines(
            image,
            LMIN=LMIN,
            LMAX=LMAX,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap,
            alpha=alpha,
            threshold=self.houghlineP_threshold,
        )
        if lines:
            # Restore image by handling horiztontal white bands
            List_images = self.split_images_lines(image, lines)
            image_restored = self.restore_image(
                image,
                List_images,
                lines,
                method=method,
                shape=image.shape[0],
            )
        else:
            image_restored = image
        return image_restored

    @staticmethod
    def _to_gray(image, method_to_gray: str = "default") -> np.array:
        """
        Convert mammography sceening to gray

        image: The image to convert
        method_to_gray: Which method to use to convert to gray (None correponds to the open cv method
        TODO explore other ways to make it gray, e.g PCA cf. Impact of Image Enhancement Module for Analysis
        of Mammogram Images for Diagnostics of Breast Cancer)

        returns: The image in grayscale
        """
        if method_to_gray == "default":
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
        threshold: int = 50,
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
            rho=1,  # Resolution of 1 pixel
            theta=np.pi / 180,  # Resolution of 1 degre
            threshold=threshold,
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
        Very experimetal, Reconstruct image with horizontal lines

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
        clahe_clipLimit: float = 2.0,
        clahe_tileGridSize: tuple = (8, 8),
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
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize
            )
            image = clahe.apply(image)
        return image

    @staticmethod
    def invert_image(image: np.array):
        """
        Invert image
        image: The screening mammography
        returns: Inverted image
        """
        return cv2.bitwise_not(image)

    @staticmethod
    def denoise(
        image: np.array,
        method: str = "NlMD",
        h: float = 3,
        block_size: int = 7,
        search_window: int = 21,
    ):
        """
        Denoise image

        image: The screening mammography
        method: Method for denoising (by default, non local means denoising)
        TODO: implement other methods of denoising

        returns: Denoised image
        """
        if method == "NlMD":
            return cv2.fastNlMeansDenoising(image, None, h, block_size, search_window)

    @staticmethod
    def gamma_correct(image: np.array, gamma: float = 2.2):
        """
        Gamma correct image

        image: The screening mammography
        gamma: The factor to use to gamma correct image

        returns: Gamma corrected image
        """
        lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, lut)
