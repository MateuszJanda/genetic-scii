#!/usr/bin/env python3

import random
import operator
from PIL import Image, ImageDraw, ImageFont
import numpy as np


STEPS = 10
POPULATION_NUM = 5
BEST_NUM = 3

IMG_WIDTH = 854
IMG_HEIGHT = 1012

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
    population = basic_population(char_width, char_height)
    orig_img = get_orig_image()

    for step in range(STEPS):
        print("Generation:", step)

        mutate(population, char_width, char_height)
        best = score(population, orig_img, char_width, char_height)
        population = corss(population, best, char_width, char_height)
        dump_best(population, best, char_width, char_height, step, 'd')

    print("End")


def singe_char_size():
    img = Image.new("L", color=BLACK, size=(IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)
    width, height = draw.textsize(text="a", font=font, spacing=FONT_SPACING)

    return width, height+FONT_SPACING


def basic_population(char_width, char_height):
    population = np.full(shape=(POPULATION_NUM, IMG_HEIGHT//char_height, IMG_WIDTH//char_width), fill_value=Char())
    return population


def get_orig_image(path="orig.png"):
    img = Image.open(path)

    assert img.size == (IMG_WIDTH, IMG_HEIGHT)
    assert img.mode == "L"

    return img


def mutate(population, char_width, char_height):
    for d in population:
        x = random.randint(0, IMG_WIDTH//char_width - 1)
        y = random.randint(0, IMG_HEIGHT//char_height - 1)
        end_x = random.randint(x + 1, IMG_WIDTH//char_width)
        end_y = random.randint(y + 1, IMG_HEIGHT//char_height)

        foreground = random.randint(0, 255)
        background = random.randint(0, 255)
        symbol = random.choice('abcd')

        d[y:end_y, x:end_x] = Char(symbol, foreground, background)


def score(population, orig_img, char_width, char_height):
    orig_img = np.array(orig_img)
    scores = {}
    for idx, dna in enumerate(population):
        img = dna_to_img(dna, char_width, char_height)
        result = np.sum(np.abs(orig_img - np.array(img)))
        scores[idx] = result

    best = sorted(scores.items(), key=operator.itemgetter(1))[:BEST_NUM]
    return [idx for idx, _ in best]


def corss(population, best, char_width, char_height):
    l = []
    for idx in range(population.shape[0]):
        l.append(np.copy(population[idx % BEST_NUM]))

    return np.array(l)


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
    for line in population:
        print("".join([ch.symbol for ch in line]))


def dump_best(population, best, char_width, char_height, step, x='x'):
    # if step % 10:
    #     return

    print(best[0])
    img = dna_to_img(population[best[0]], char_width, char_height)
    img.save(x + str(step) + ".png")



if __name__ == '__main__':
    main()
