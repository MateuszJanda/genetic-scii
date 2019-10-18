#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont


STEPS = 1
SPEC_CNT = 100

WIDTH = 400
HEIGHT = 400

BLACK = 0
WHITE = 255

FONT_NAME = 'UbuntuMono-R'
FONT_SIZE = 17



def mutate():
    dna_to_img()
    pass


def score():
    pass


def corss():
    pass


def dna_to_img():
    img = Image.new("L", color=BLACK, size=(WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)

    text = "to jest tekst\n   i trzy spacje"

    draw.text(xy=(0, 0), text=text, font=font, fill=WHITE)
    img.save("out.png")


def main():

    for step in range(STEPS):
        print("Generation:", step)

        mutate()

        score()

        corss()


if __name__ == '__main__':
    main()
