# geneti-scii

Generate ASCII/Unicode (glitch) image with genetic algorithm.

# Media source

- https://commons.wikimedia.org/wiki/File:The-punisher-logo-png-transparent.png
- https://commons.wikimedia.org/wiki/File:Pirate_Flag_of_John_Taylor.svg
- https://commons.wikimedia.org/wiki/File:Crowned_Skull.svg

## Requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ImageMagic cheatsheet:

Reduced just enough so as to best fit into the given size
```bash
convert input.jpg -resize 64x64 output.jpg
```

For new size (ignore aspect ratio)
```bash
convert input.jpg -resize 400x400\! output.jpg
```

Change RGB to grayscale
```bash
convert input.jpg -set colorspace Gray -separate -average output.jpg
```

Convert JPEG format to PNG
```bash
convert input.jpg output.png
```