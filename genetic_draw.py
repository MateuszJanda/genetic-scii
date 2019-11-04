#!/usr/bin/env python3

import random
import time
import copy
import operator
from PIL import Image, ImageDraw, ImageFont
import numpy as np


STEPS = 21
POPULATION_NUM = 100
BEST_NUM = 3

BLACK = 0
WHITE = 255

# FONT_NAME = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONT_NAME = 'DejaVuSansMono'
FONT_SIZE = 16
FONT_SPACING = 2

CHAR_BASE_BASIC = 'asdf'


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
        tic = time.time()

        mutate(population, char_shape, CHAR_BASE_BASIC)
        best = scores(population, orig_img, char_shape)
        population = corss(population, best)
        dump_best(population, best, orig_img.shape, char_shape, step)

        print("Generation:", step, "time:", time.time() - tic)

    print("End")


def singe_char_shape():
    img = Image.new("L", color=BLACK, size=(50, 50))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)
    width, height = draw.textsize(text="a", font=font, spacing=FONT_SPACING)

    return height+FONT_SPACING, width


def basic_population(img_shape, char_shape):
    img = Image.new("L", color=BLACK, size=(img_shape[1], img_shape[0]))
    dna = np.full(shape=(img_shape[0]//char_shape[0], img_shape[1]//char_shape[1]), fill_value=Char())

    population = [(np.copy(dna), copy.copy(img)) for _ in range(POPULATION_NUM)]
    return population


def get_orig_img(path="orig.png"):
    img = Image.open(path)

    assert img.mode == "L"

    return np.array(img)


def mutate(population, char_shape, char_base):
    font = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)

    for dna, img in population:
        width, height = img.size
        begin_x = random.randint(0, width//char_shape[1] - 1)
        begin_y = random.randint(0, height//char_shape[0] - 1)
        end_x = random.randint(begin_x + 1, width//char_shape[1])
        end_y = random.randint(begin_y + 1, height//char_shape[0])

        foreground = random.randint(0, 255)
        background = random.randint(0, 255)
        symbol = random.choice(char_base)

        draw = ImageDraw.Draw(img)

        for x in range(begin_x, end_x):
            for y in range(begin_y, end_y):
                char = dna[y, x]
                f = (char.foreground + foreground)//2
                b = (char.background + background)//2
                dna[y, x] = Char(symbol, f, b)

                pos_x, pos_y = x*char_shape[1], y*char_shape[0]
                draw.rectangle(xy=[(pos_x, pos_y), (pos_x + char_shape[1], pos_y + char_shape[0])], fill=char.background)
                draw.text(xy=(pos_x, pos_y), text=char.symbol, fill=char.foreground, font=font, spacing=FONT_SPACING)


def scores(population, orig_img, char_shape):
    scores = {}
    for idx, val in enumerate(population):
        dna, img = val
        result = np.sum(np.abs(orig_img - img))
        scores[idx] = result

    best = sorted(scores.items(), key=operator.itemgetter(1))[:BEST_NUM]
    return [idx for idx, _ in best]


def corss(population, best):
    result = [np.copy(population[idx % BEST_NUM]) for idx in range(len(population))]
    return result


def dna_to_img(dna, img_shape, char_shape):
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
    if step % 10:
        return

    img = dna_to_img(population[best[0]][0], img_shape, char_shape)
    img.save("f" + str(step) + ".png")


if __name__ == '__main__':
    main()
