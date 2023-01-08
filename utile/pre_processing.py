import cv2
import numpy as np
from typing import List
import torch

# from skimage.metrics import structural_similarity


class PreProcessingPipeline(object):
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
        pectoral_muscle: bool = True,
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
        self.pectoral_muscle = pectoral_muscle

        # Default values for preprocessing that you can change in calling the constructor of the Pipeline class
        # Methods to adopt in the pipeline for each step
        self.denoising_method = "NlMD"
        self.normalization_method = "clahe"
        self.remove_line_method = "delete"
        self.pectoral_muscle_method = "prewitt"

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

        # Arguments for pectoral muscle removal
        self.pectoral_muscle_kernel_size_closing = (5, 5)
        self.pectoral_muscle_thresh_mask_edges = 0.95
        self.pectoral_muscle_kernel_erosion_shape = (1, 2)

        # Possibilty to adjust parameters
        self.__dict__.update(methods_args)

    def __call__(self, sample: dict):
        """
        Preprocess function for Pytorch pipeline
        """
        sample["image"] = torch.tensor(
            np.reshape(
                np.array(self.pre_process(sample["image"]), dtype=np.float32),
                (1, sample["image"].shape[0], sample["image"].shape[1]),
            )
        )
        sample["features"] = torch.tensor(
            np.array(sample["features"], dtype=np.float32)
        )
        sample["labels"] = torch.tensor(np.array(sample["labels"], dtype=np.float32))
        return sample

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

            # Convert image to gray
            image = self._to_gray(image, method_to_gray=self.method_to_gray)
            shape = image.shape
            # This part is to remove small black bands at top and bottom of image

            contours = self.get_contours(
                image, thresh_low=self.contours_low, thresh_high=self.contours_high
            )
            biggest_contour = max(contours, key=cv2.contourArea)
            # Apply mask on biggest contour
            mask = self.draw_contours(contours, image, biggest=True)[1]
            # Remove small spaces at the top and bottom of image
            _, y, _, h = cv2.boundingRect(biggest_contour)
            image = image[y + 2 : y + h, :]
            # Resize image to correct shape
            image = cv2.resize(image, shape)

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
            if self.pectoral_muscle:
                image = self._remove_pectoral_muscle(
                    image,
                    method=self.pectoral_muscle_method,
                    kernel_size_closing=self.pectoral_muscle_kernel_size_closing,
                    thresh_mask_edges=self.pectoral_muscle_thresh_mask_edges,
                    kernel_erosion_shape=self.pectoral_muscle_kernel_erosion_shape,
                )
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
    ) -> np.array:
        """
        Remove annotation in image

        image: The screening mammography

        returns: Image without annotation
        """
        shape = image.shape
        # Get contours
        contours = self.get_contours(
            image, thresh_low=thresh_low, thresh_high=thresh_high
        )
        biggest_contour = max(contours, key=cv2.contourArea)
        # Apply mask on biggest contour
        mask = self.draw_contours(contours, image, biggest=True)[1]
        # # Remove small spaces at the top and bottom of image
        # _, y, _, h = cv2.boundingRect(biggest_contour)
        # image = image[y + 50 : y + h, :]
        # # Resize image to correct shape
        # image = cv2.resize(image, shape)
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
    ) -> np.array:
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

    def _remove_pectoral_muscle(
        self,
        image: np.array,
        method: str = "prewitt",
        kernel_size_closing: tuple = (5, 5),
        thresh_mask_edges: float = 0.95,
        kernel_erosion_shape: tuple = (1, 2),
    ) -> np.array:
        """
        Method to remove pectoral muscle (implementation adapted from
        Removal of pectoral muscle based on topographic map and shape-shifting silhouette)

        image: Screening mammography
        method: The method for edge detection (here prewitt method is used)
        kernel_size_closing: Kernel used for closing method (cf. opencv)
        thresh_mask_edges: Adaptative relative thresholding for mask
        kernel_erosion_shape: Kernel of the erosion

        returns: Image without pectoral muscle
        """
        # Get edges
        edges = self.get_edges(image=image, method=method)
        edges = self.remove_useless_edges(
            edges=edges,
            kernel_size_closing=kernel_size_closing,
            thresh_mask_edges=thresh_mask_edges,
            kernel_erosion_shape=kernel_erosion_shape,
        )
        hull = self.get_convex_hull(edges=edges)
        mask = np.zeros_like(image)

        # Fill the convex hull with 1's in the mask
        cv2.fillConvexPoly(mask, hull, 1)

        # Apply the binary mask to the image
        image = cv2.bitwise_and(image, image, mask=mask)

        return image

    @staticmethod
    def get_edges(image: np.array, method: str = "prewitt") -> np.array:
        """
        Detect edges from an image given a method

        image: Screening mammography
        method: The method for edge detection (here prewitt method is used)

        returns: Edges of the image
        """
        edges = []
        image = image / 255
        if method == "prewitt":
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

            img_prewittx = cv2.filter2D(image, -1, kernelx)
            img_prewitty = cv2.filter2D(image, -1, kernely)

            # Calculate the gradient magnitude
            edges = np.sqrt(np.square(img_prewittx) + np.square(img_prewitty))

            # Normalize the gradient magnitude image
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        return edges

    @staticmethod
    def remove_useless_edges(
        edges: np.array,
        kernel_size_closing: tuple = (5, 5),
        thresh_mask_edges: float = 0.95,
        kernel_erosion_shape: tuple = (1, 2),
    ) -> np.array:
        """
        Remove useless edges by adaptive thresholding and small erosion

        edges: Detected edges from image
        kernel_size_closing: Kernel used for closing method (cf. opencv)
        thresh_mask_edges: Adaptative relative thresholding for mask
        kernel_erosion_shape: Kernel of the erosion

        returns: Filtered edges
        """
        # Define the kernel for the closing operation
        kernel = np.ones(kernel_size_closing, np.uint8)

        # Apply the closing operation to the image
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        intensities = closed.flatten()

        # Create a new array of non-zero intensities
        intensities = intensities[intensities > 10]

        # Sort the array of pixel intensities
        intensities.sort()

        # Find the index of the thresh_mask_edges quantile
        index = int(len(intensities) * thresh_mask_edges)

        # Retrieve the 50th quantile value from the sorted array
        quantile = intensities[index]

        _, edges_thresh = cv2.threshold(closed, quantile, 255, cv2.THRESH_BINARY)

        # Define the kernel for the erosion operation
        kernel = np.ones(kernel_erosion_shape, np.uint8)

        # Apply the erosion operation to the image
        edges_thresh = cv2.erode(edges_thresh, kernel, iterations=1)
        return edges_thresh

    @staticmethod
    def get_convex_hull(edges: np.array) -> np.array:
        """
        Get convex hull from edges of image

        edges: Detected edges from image

        returns: Minimum convex hull (cf. opencv)
        """
        # Find the non-zero pixels in the image
        points = np.argwhere(edges > 0)

        points = np.array([[elem[1], elem[0]] for elem in points])

        # Calculate the convex hull of the points
        hull = cv2.convexHull(points)
        return hull

    @staticmethod
    def draw_convex_hull(hull: np.array, image: np.array) -> np.array:
        """
        Draw convex hull on an image (cf opencv)

        image: The image to convert
        hull: The hull to be drawn

        returns: The image with drawn hull in red
        """
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Draw the convex hull on the image
        image = cv2.polylines(image, [hull], True, (255, 0, 0), 2)
        return image

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
    ) -> tuple:
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
    ) -> np.array:
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
    ) -> np.array:
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
    def invert_image(image: np.array) -> np.array:
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
    ) -> np.array:
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
    def gamma_correct(image: np.array, gamma: float = 2.2) -> np.array:
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

    @staticmethod
    def evaluate_metrics(
        raw_imgs: List[np.array] or np.array,
        pre_processed_imgs: List[np.array] or np.array,
    ) -> tuple:
        """
        Evaluate different metrics for pre_processing

        raw_imgs: Raw image or list of raws images
        pre_processed_imgs: pre-processed image or list of pre-processed images

        returns: SNR and SSIM between each pair of raw and pre-processed images
        """
        snrs = []
        ssims = []

        if not (isinstance(raw_imgs, type(pre_processed_imgs))):
            raise Exception("Please provide either two lists of images or two images")
        if isinstance(raw_imgs, List):
            if not len(raw_imgs) == len(pre_processed_imgs):
                raise Exception(
                    "Please provide the same number of images (one raw image and associated pre-processed image)"
                )
        if not isinstance(raw_imgs, List):
            raw_imgs = [raw_imgs]
            pre_processed_imgs = [pre_processed_imgs]

        for raw_img, pre_processed_img in zip(raw_imgs, pre_processed_imgs):

            # Calculate the difference image
            diff = pre_processed_img - raw_img

            # Calculate the mean and standard deviation of the difference image
            mean = np.mean(diff)
            std = np.std(diff)

            # Calculate the SNR
            snr = mean / std
            snrs.append(snr)

            # TODO handle this, new problem with skimage
            # (score, _) = structural_similarity(raw_img, pre_processed_img, full=True)
            score = 0
            ssims.append(score)

        return snrs, ssims
