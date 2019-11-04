#!/usr/bin/env python3

import random
import operator
from PIL import Image, ImageDraw, ImageFont
import numpy as np


STEPS = 2
POPULATION_NUM = 5
BEST_NUM = 3

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
    char_shape = singe_char_shape()
    orig_img = get_orig_img()
    population = basic_population(orig_img.shape, char_shape)

    for step in range(STEPS):
        print("Generation:", step)

        mutate(population, orig_img.shape, char_shape)
        best = scores(population, orig_img, char_shape)
        population = corss(population, best)
        dump_best(population, best, orig_img.shape, char_shape, step)

    print("End")


def singe_char_shape():
    img = Image.new("L", color=BLACK, size=(50, 50))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)
    width, height = draw.textsize(text="a", font=font, spacing=FONT_SPACING)

    return height+FONT_SPACING, width


def basic_population(img_shape, char_shape):
    population = np.full(shape=(POPULATION_NUM, img_shape[0]//char_shape[0],
        img_shape[1]//char_shape[1]), fill_value=Char())
    return population


def get_orig_img(path="orig.png"):
    img = Image.open(path)

    assert img.mode == "L"

    return np.array(img)


def mutate(population, img_shape, char_shape):
    for dna in population:
        x = random.randint(0, img_shape[1]//char_shape[1] - 1)
        y = random.randint(0, img_shape[0]//char_shape[0] - 1)
        end_x = random.randint(x + 1, img_shape[1]//char_shape[1])
        end_y = random.randint(y + 1, img_shape[0]//char_shape[0])

        foreground = random.randint(0, 255)
        background = random.randint(0, 255)
        symbol = random.choice('abcd')

        dna[y:end_y, x:end_x] = Char(symbol, foreground, background)


def scores(population, orig_img, char_shape):
    scores = {}
    for idx, dna in enumerate(population):
        img = dna_to_img(dna, orig_img.shape, char_shape)
        result = np.sum(np.abs(orig_img - np.array(img)))
        scores[idx] = result

    best = sorted(scores.items(), key=operator.itemgetter(1))[:BEST_NUM]
    return [idx for idx, _ in best]


def corss(population, best):
    l = []
    for idx in range(population.shape[0]):
        l.append(np.copy(population[idx % BEST_NUM]))

    return np.array(l)


def dna_to_img(dna, img_shape, char_shape):
    print(img_shape[1], img_shape[0])
    img = Image.new("L", color=BLACK, size=(img_shape[1], img_shape[0]))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)

    for y, line in enumerate(dna):
        for x, char in enumerate(line):
            pos_x, pos_y = x*char_shape[1], y*char_shape[0]
            draw.rectangle(xy=[(pos_x, pos_y), (pos_x + char_shape[1], pos_y + char_shape[0])], fill=char.background)
            draw.text(xy=(pos_x, pos_y), text=char.symbol, fill=char.foreground, font=font, spacing=FONT_SPACING)

    return img


def print_dna(dna):
    for line in population:
        print("".join([ch.symbol for ch in line]))


def dump_best(population, best, img_shape, char_shape, step):
    # if step % 10:
    #     return

    img = dna_to_img(population[best[0]], img_shape, char_shape)
    img.save("d" + str(step) + ".png")


if __name__ == '__main__':
    main()
