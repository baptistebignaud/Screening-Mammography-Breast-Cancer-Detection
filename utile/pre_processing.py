import cv2
import numpy as np
from typing import List


class PreProcessingPipeline:
    def __init__(
        self,
        remove_annotation=True,
        remove_line=False,
        normalization="clahe",
        denoising=True,
        gamma_correction=False,
        background_uniformity=False,
        sharpening=False,
        normalisation=False,
    ) -> None:
        self.remove_annotation = remove_annotation
        self.remove_line = remove_line
        self.normalization = normalization
        self.denoising = denoising
        self.gamma_correction = gamma_correction
        self.background_uniformity = background_uniformity
        self.sharpening = sharpening
        self.normalisation = normalisation

    @staticmethod
    def _to_gray(image, method=None):
        if not method:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def get_contours(image):
        # Perform thresholding to create a binary image
        _, binary = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def draw_contours(contours, image, biggest=True):
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

    def _remove_annotation(self, image):
        contours = self.get_contours(image)
        mask = self.draw_contours(contours, image, biggest=True)[1]
        return cv2.bitwise_and(image, mask)

    @staticmethod
    def get_horizontal_lines(
        image, LMIN=400, LMAX=500, minLineLength=100, maxLineGap=20, alpha=0.1
    ):

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
    def split_images_lines(image, filtered_lines):
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
        image, list_images, filtered_lines, method="delete", thresh_low=0.1, shape=512
    ):
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

    def pre_process(self):
        pass
