#!/usr/bin/env python3

import random
import time
import copy
import string
from PIL import Image, ImageDraw, ImageFont
import numpy as np


CHAR_BASE_BASIC = 'asdf'
CHAR_BASE_ASCII = string.digits + string.ascii_letters + string.punctuation

STEPS = 5000
POPULATION_NUM = 200
BEST_NUM = 3
MUTATION_FACTOR = 0.1

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
    return height + FONT_SPACING, width

CHAR_SHAPE = singe_char_shape()



class Char:
    def __init__(self, symbol=" ", foreground=WHITE, background=BLACK):
        self.symbol = symbol
        self.foreground = foreground
        self.background = background


def main():
    random.seed(1321)

    orig_arr = get_orig_array()
    orig_arr = convert_to_mosaic(orig_arr)
    population = basic_population(orig_arr.shape)

    for step in range(STEPS):
        tic = time.time()

        mutate(population, CHAR_BASE_BASIC)
        best_idx, scores = select(population, orig_arr)
        population = cross(population, best_idx)

        if step % 10 == 0:
            dump_img(population, best_idx[0], step)

        print("Generation:", step, "time:", time.time() - tic, "best_idx:", scores[best_idx[0]])

    print("End")


def basic_population(img_shape):
    img = Image.new("L", color=BLACK, size=(img_shape[1], img_shape[0]))
    dna = np.full(shape=(img_shape[0]//CHAR_SHAPE[0], img_shape[1]//CHAR_SHAPE[1]), fill_value=Char())
    population = [(np.copy(dna), copy.copy(img)) for _ in range(POPULATION_NUM)]
    return population


def get_orig_array(path="orig.png"):
    img = Image.open(path)

    assert img.mode == "L"

    return np.array(img)


def convert_to_mosaic(arr):
    for x in range(0, arr.shape[1], CHAR_SHAPE[1]):
        for y in range(0, arr.shape[0], CHAR_SHAPE[0]):
            arr[y:y+CHAR_SHAPE[0], x:x+CHAR_SHAPE[1]] = np.average(arr[y:y+CHAR_SHAPE[0], x:x+CHAR_SHAPE[1]])

    return arr


def mutate(population, char_base, random_background=True):
    for dna, img in population:
        width = img.size[0]//CHAR_SHAPE[1]
        height = img.size[1]//CHAR_SHAPE[0]
        begin_x = random.randint(0, width - 1)
        begin_y = random.randint(0, height - 1)
        size_x = int(random.randint(1, width) * MUTATION_FACTOR)
        size_y = int(random.randint(1, height) * MUTATION_FACTOR)

        # symbol = random.choice(char_base)
        symbol = ' '
        # new_foreground = random.randint(0, 255)
        new_foreground = 0
        # if random_background:
        new_background = random.randint(0, 255)
        # else:
            # new_background = 0

        draw = ImageDraw.Draw(img)

        for x in range(begin_x, min(begin_x + size_x, width)):
            for y in range(begin_y, min(begin_y + size_y, height)):
                char = dna[y, x]
                foreground = (char.foreground + new_foreground)//2
                background = (char.background + new_background)//2
                dna[y, x] = Char(symbol, foreground, background)

                draw_char(draw, x, y, dna[y, x])


def select(population, orig_arr):
    scores = {}
    for idx, (_, img) in enumerate(population):
        result = np.sum((orig_arr - img)**2)
        scores[idx] = result

    best_idx = sorted(scores, key=scores.get)[:BEST_NUM]

    return best_idx, scores


def cross(population, best_idx):
    best_specimens = [copy.deepcopy(population[idx]) for idx in best_idx]

    result = []
    for idx in range(len(population)):
        dna, img = best_specimens[idx % BEST_NUM]
        result.append((np.copy(dna), copy.copy(img)))

    return result


def print_dna(dna):
    for line in population:
        print("".join([ch.symbol for ch in line]))


def dump_img(population, idx, step):
    dna, img = population[idx]
    img_shape = (img.size[1], img.size[0])
    out_img = dna_to_img(dna, img_shape)
    out_img.save("a%04d.png" % step)

    # assert np.all(np.array(out_img) == np.array(img))


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


if __name__ == '__main__':
    main()
