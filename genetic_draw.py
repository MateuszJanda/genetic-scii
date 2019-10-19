#!/usr/bin/env python3

import random
from PIL import Image, ImageDraw, ImageFont


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
    def __init__(self):
        self.foreground = WHITE
        self.background = BLACK
        self.symbol = "a"


def main():
    char_width, char_height = singe_char_size()
    dna = basic_dna(char_width, char_height)

    for step in range(STEPS):
        print("Generation:", step)

        mutate(dna, char_width, char_height)

        score()

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


def mutate(dna, char_width, char_height):
    x = random.randint(0, IMG_WIDTH//char_width - 1)
    y = random.randint(0, IMG_HEIGHT//char_height - 1)
    width = random.randint(1, IMG_WIDTH//char_width - x)
    height = random.randint(1, IMG_HEIGHT//char_height - y)
    char = random.choice('abcd')

    img = dna_to_img(dna[0], char_width, char_height)
    img.save("out.png")


def score():
    pass


def corss():
    pass


if __name__ == '__main__':
    main()
