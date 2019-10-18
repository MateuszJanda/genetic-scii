#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont
import numpy as np


STEPS = 1000
SPEC_CNT = 100

WIDTH = 100
HEIGHT = 100

BLACK = 0
WHITE = 255

FONT_NAME = 'UbuntuMono-R'
FONT_SIZE = 17



def mutate():
    pass


def score():
    pass


def corss():
    pass


def dna_to_img():
    pil_img = Image.new('RGB', color=BLACK, size=(WIDTH, HEIGHT))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)

    draw.text(xy=(0, 0), text=text, font=font, fill=WHITE)
    terminal_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():

    for step in range(STEPS):
        print("Generation:", step)

        mutate()

        score()

        corss()


if __name__ == '__main__':
    main()
