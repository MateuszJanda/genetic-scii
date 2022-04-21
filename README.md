# geneti-scii
Generate ASCII/Unicode (glitch) image with genetic algorithm.

## Requirements
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ImageMagic cheatsheet:

Reduced just enough so as to best fit into the given size
```
convert input.jpg -resize 64x64 output.jpg
```

For new size (ignore aspect ratio)
```
convert input.jpg -resize 400x400\! output.jpg
```

Change RGB to grayscale
```
convert input.jpg -set colorspace Gray -separate -average output.jpg
```

Convert JPEG format to PNG
```
convert input.jpg output.png
```