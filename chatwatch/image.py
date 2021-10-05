from collections import Iterator
from datetime import time
from enum import Enum
from typing import List

from PIL import Image, ImageDraw
from PIL import ImageFilter, ImageOps
import pytesseract
import cv2
import numpy as np
import time
import io
import os
from google.cloud import vision
import cachetools.func



# open the "original" image

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


class ChatParser:
    def __init__(self, filename):
        self.filename = filename
        self.client = vision.ImageAnnotatorClient()


    def read(self, filename=None) -> Image:
        return Image.open(self.filename if not filename else filename)

    def preprocess(self, image: Image, top=0, left=0, right=0, bottom=0) -> Image:
        width, height = image.size
        print(width, height)
        #2592x1944
        left = left
        top = top
        right = width + right
        bottom = height + bottom
        display = image.crop((left, top, right, bottom))
        display.save('cropped.png')
        smoothed = display.filter(ImageFilter.UnsharpMask)
        smoothed.save('processed.png')
        return 'processed.png'

    # def cv2_process(self, filename):
    #     img = cv2.imread(filename)
    #     (h, w) = img.shape[:2]
    #     img = cv2.resize(img, (w, h))
    #     gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     thr = cv2.threshold(gry, 5, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_MASK)[1]
    #     # Inverting
    #     gray = 255 - gry
    #     emp = np.full_like(gray, 255)
    #     emp -= gray
    #
    #     # Thresholding
    #     emp[emp == 0] = 255
    #     emp[emp < 100] = 0
    #     final = Image.fromarray(thr)
    #     final.save("final.png")
    #     print("Saved final output")
    #     return "final.png"

    def parse(self, top: int = 900, left: int = 950, right=-592, bottom=-650):
        image = self.read()
        smooth_im = self.preprocess(image, top, left, right, bottom)

    def draw_boxes(self, image, bounds, color):
        """Draw a border around the image using the hints in the vector list."""
        draw = ImageDraw.Draw(image)

        for bound in bounds:
            draw.polygon([
                bound.vertices[0].x, bound.vertices[0].y,
                bound.vertices[1].x, bound.vertices[1].y,
                bound.vertices[2].x, bound.vertices[2].y,
                bound.vertices[3].x, bound.vertices[3].y], None, color)
        return image

    def get_document_bounds(self, document, feature) -> Iterator[str]:
        bounds = []
        # Collect specified feature bounds by enumerating all document features
        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    words = []
                    for word in paragraph.words:
                        symb = []
                        for symbol in word.symbols:
                            if (feature == FeatureType.SYMBOL):
                                bounds.append(symbol.bounding_box)
                                symb.append(symbol.text)
                        words.append("".join(symb))
                        if (feature == FeatureType.WORD):
                            bounds.append(word.bounding_box)
                    yield(" ".join(words))
                    if (feature == FeatureType.PARA):
                        bounds.append(paragraph.bounding_box)

                if (feature == FeatureType.BLOCK):
                    bounds.append(block.bounding_box)

        # The list `bounds` contains the coordinates of the bounding boxes.
        return bounds

    @cachetools.func.ttl_cache(maxsize=128, ttl=3*60)
    def get_document(self, image_filename):
        file_name = os.path.relpath(image_filename)

        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            print(f"Opening image at {file_name}")
            content = image_file.read()

        image = vision.Image(content=content)
        print("Loading to google....")

        # Performs label detection on the image file
        response = self.client.document_text_detection(image=image)
        document = response.full_text_annotation
        return document

    @staticmethod
    def flatten(t):
        return [item for sublist in t for item in sublist]

    def get_last_n_sentences(self, n=2) -> str:
        im = Image.open(self.filename)
        processed = self.preprocess(im, 400, 800, 0, 0)
        doc = self.get_document(processed)
        sentences = self.get_document_bounds(doc, FeatureType.SYMBOL)

        try:
            last_sentence = ". ".join(list(sentences)[-n:])
        except IndexError:
            last_sentence = list(sentences)[-1]
        return last_sentence


def handle_new_message(message: str):
    from twilio.rest import Client
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)

    message = client.messages \
        .create(
        body=message,
        from_='+15703644920',
        to='+18054412653'
    )



if __name__ == "__main__":
    last_seen = ""
    previous_last_seen = ""
    target = "Mindi McDowell"
    while True:
        p = ChatParser("/Users/grice/repos/chatwatch-pi/image4.jpg")
        last_seen_sentence = p.get_last_n_sentences(n=10)
        if last_seen_sentence:
            last_seen_processed = last_seen_sentence.lstrip(" ")
            if target_idx := last_seen_processed.find(target):
                last_seen_processed = last_seen_processed[target_idx:]
                if last_seen_processed != previous_last_seen:
                    print(f"Target {target} found; showing sentence containing {target}:")
                    previous_last_seen = last_seen_processed
                    print(f"New message from {target}:\n{last_seen_processed}")
                    handle_new_message(last_seen_processed)
        print("Sleeping for 120 seconds to await new texts")
        time.sleep(120)
