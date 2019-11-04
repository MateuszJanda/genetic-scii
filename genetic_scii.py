#!/usr/bin/env python3

import random
import time
import copy
import operator
from PIL import Image, ImageDraw, ImageFont
import numpy as np


CHAR_BASE_BASIC = 'asdf'

STEPS = 10
POPULATION_NUM = 2
BEST_NUM = 3

BLACK = 0
WHITE = 255

FONT_NAME = 'DejaVuSansMono'
FONT_SIZE = 16
FONT_SPACING = 2
FONT = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)


def singe_char_shape():
    img = Image.new("L", color=BLACK, size=(50, 50))
    draw = ImageDraw.Draw(img)
    width, height = draw.textsize(text="a", font=FONT, spacing=FONT_SPACING)
    return height+FONT_SPACING, width

CHAR_SHAPE = singe_char_shape()



class Char:
    def __init__(self, symbol=" ", foreground=WHITE, background=BLACK):
        self.symbol = symbol
        self.foreground = foreground
        self.background = background


def main():
    random.seed(1321)

    orig_img = get_orig_img()
    population = basic_population(orig_img.shape)

    for step in range(STEPS):
        tic = time.time()

        mutate(population, CHAR_BASE_BASIC)
        best = scores(population, orig_img)
        population = corss(population, best)
        dump_best(population, best, orig_img.shape, step)

        print("Generation:", step, "time:", time.time() - tic)

    print("End")


def basic_population(img_shape):
    img = Image.new("L", color=BLACK, size=(img_shape[1], img_shape[0]))
    dna = np.full(shape=(img_shape[0]//CHAR_SHAPE[0], img_shape[1]//CHAR_SHAPE[1]), fill_value=Char())

    population = [(np.copy(dna), copy.copy(img)) for _ in range(POPULATION_NUM)]
    return population


def get_orig_img(path="orig.png"):
    img = Image.open(path)

    assert img.mode == "L"

    return np.array(img)


def mutate(population, char_base):
    for dna, img in population:
        width, height = img.size
        begin_x = random.randint(0, width//CHAR_SHAPE[1] - 1)
        begin_y = random.randint(0, height//CHAR_SHAPE[0] - 1)
        end_x = random.randint(begin_x + 1, width//CHAR_SHAPE[1]) // 3
        end_y = random.randint(begin_y + 1, height//CHAR_SHAPE[0]) // 3

        new_foreground = random.randint(0, 255)
        new_background = random.randint(0, 255)
        symbol = random.choice(char_base)

        draw = ImageDraw.Draw(img)

        for x in range(begin_x, end_x):
            for y in range(begin_y, end_y):
                char = dna[y, x]
                foreground = (char.foreground + new_foreground)//2
                background = (char.background + new_background)//2
                dna[y, x] = Char(symbol, foreground, background)

                draw_char(draw, x, y, dna[y, x])


def scores(population, orig_img):
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


def dna_to_img(dna, img_shape):
    img = Image.new("L", color=BLACK, size=(img_shape[1], img_shape[0]))
    draw = ImageDraw.Draw(img)

    for y, line in enumerate(dna):
        for x, char in enumerate(line):
            draw_char(draw, x, y, char)

    return img


def draw_char(draw, x, y, char):
    pos_x, pos_y = x*CHAR_SHAPE[1], y*CHAR_SHAPE[0]
    draw.rectangle(xy=[(pos_x, pos_y), (pos_x + CHAR_SHAPE[1] - 1, pos_y + CHAR_SHAPE[0] - 1)], fill=char.background)
    draw.text(xy=(pos_x, pos_y), text=char.symbol, fill=char.foreground, font=FONT, spacing=FONT_SPACING)


def print_dna(dna):
    for line in population:
        print("".join([ch.symbol for ch in line]))


def dump_best(population, best, img_shape, step):
    if step % 10:
        return

    img = dna_to_img(population[best[0]][0], img_shape)
    img.save("g%04d.png" % step)

    assert np.all(np.array(img) == np.array(population[best[0]][1]))


if __name__ == '__main__':
    main()
