#!/usr/bin/env python3

import random
from PIL import Image, ImageDraw, ImageFont


STEPS = 1
SPEC_CNT = 100

WIDTH = 400
HEIGHT = 400

BLACK = 0
WHITE = 255

# FONT_NAME = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONT_NAME = 'DejaVuSansMono'
FONT_SIZE = 16
FONT_SPACING = 2


def mutate(dna):
    # dna_to_img()
    x = random.randint(0, WIDTH//FONT_SIZE//2 - 1)
    y = random.randint(0, HEIGHT//FONT_SIZE - 1)
    width = random.randint(1, WIDTH//FONT_SIZE//2 - x)
    char = random.choice('abcd')

    dna_to_img()


def score():
    pass


def corss():
    pass


def dna_to_img():
    img = Image.new("L", color=BLACK, size=(WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)

    text = "to jest tekst\n   i trzy spacje"
    text = "this text is to looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong"

    # print(font.getsize("a"))
    print(draw.textsize("a", font))
    print(draw.textsize("asdf\nasdf", font))
    print(font.path)
    # print(draw.multiline_textsize("asdf\nasdf", font))

    # draw.text(xy=(0, 0), text=text, font=font, fill=WHITE)

    # default spacing 4

    draw.text(xy=(0, 0), text="╣ ╠╣ ░ ▅\n╣ ╣ ░ ▅\ncommand not found", fill=WHITE, font=font, spacing=FONT_SPACING)
    # draw.text(xy=(0, 15+4), text="asdf", font=font, fill=WHITE)
    # draw.text((0,0), text="asdf\nasdf", font=font, fill=WHITE)
    # draw.text(xy=(0, 0), text="xxx\njkl", font=font, fill=WHITE)

    # draw.multiline_text((0,0), text="asdf\nasdf", font=font, fill=WHITE)

    img.save("out.png")


def main():
    dna = []
    for _ in range(SPEC_CNT):
        dna.append([" " for _ in range(WIDTH//FONT_SIZE//2)] for _ in range(HEIGHT//FONT_SIZE))


    for step in range(STEPS):
        print("Generation:", step)

        mutate(dna)

        score()

        corss()


if __name__ == '__main__':
    main()
