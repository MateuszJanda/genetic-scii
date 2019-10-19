#!/usr/bin/env python3

import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np


STEPS = 1
SPEC_CNT = 100

IMG_WIDTH = 400
IMG_HEIGHT = 400

BLACK = 0
WHITE = 255

# FONT_NAME = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONT_NAME = 'DejaVuSansMono'
FONT_SIZE = 16
FONT_SPACING = 2


class Char:
    def __init__(self, symbol=" ", foreground=WHITE, background=BLACK):
        self.symbol = symbol
        self.foreground = foreground
        self.background = background


def main():
    char_width, char_height = singe_char_size()
    dna = basic_dna(char_width, char_height)
    orig_img = get_orig_image()

    for step in range(STEPS):
        print("Generation:", step)

        mutate(dna, char_width, char_height)

        score(dna, orig_img, char_width, char_height)

        corss()


def singe_char_size():
    img = Image.new("L", color=BLACK, size=(IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)
    width, height = draw.textsize(text="a", font=font, spacing=FONT_SPACING)

    return width, height+FONT_SPACING


def basic_dna(char_width, char_height):
    dna = []
    for _ in range(SPEC_CNT):
        dna.append([[Char() for _ in range(IMG_WIDTH//char_width)] for _ in range(IMG_HEIGHT//char_height)])

    return dna


def get_orig_image(path="orig.png"):
    img = Image.open(path)

    assert img.size == (IMG_WIDTH, IMG_HEIGHT)
    assert img.mode == "L"

    return img


def mutate(dna, char_width, char_height):
    for d in dna:
        x = random.randint(0, IMG_WIDTH//char_width - 1)
        y = random.randint(0, IMG_HEIGHT//char_height - 1)
        end_x = random.randint(x + 1, IMG_WIDTH//char_width)
        end_y = random.randint(y + 1, IMG_HEIGHT//char_height)

        foreground = random.randint(0, 255)
        background = random.randint(0, 255)
        symbol = random.choice('abcd')

        for row in range(y, end_y):
            for col in range(x, end_x):
                d[row][col] = Char(symbol, foreground, background)


def score(dna, orig_img, char_width, char_height):
    scores = {}
    for idx, d in enumerate(dna):
        img = dna_to_img(d, char_width, char_height)
        result = np.sum(np.abs(np.array(orig_img) - np.array(img)))
        scores[idx] = result

    return scores


def corss():
    pass


def dna_to_img(dna, char_width, char_height):
    img = Image.new("L", color=BLACK, size=(IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)

    for y, line in enumerate(dna):
        for x, char in enumerate(line):
            pos_x, pos_y = x*char_width, y*char_height
            draw.rectangle(xy=[(pos_x, pos_y), (pos_x + char_width, pos_y + char_height)], fill=char.background)
            draw.text(xy=(pos_x, pos_y), text=char.symbol, fill=char.foreground, font=font, spacing=FONT_SPACING)

    return img


def print_dna(dna):
    for line in dna:
        print("".join([ch.symbol for ch in line]))


if __name__ == '__main__':
    main()
