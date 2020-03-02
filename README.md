## Requirements
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```
Tested with:
- Pillow           6.2.0
- numpy            1.17.3

## ImageMagic cheatsheet:

Reduced just enough so as to best fit into the given size
```
convert orig.jpg -resize 64x64 shrink.jpg
```

For new size (ignore aspect ratio)
```
convert orig.jpg -resize 400x400\! shrink.jpg
```

Change RGB to grayscale
```
convert orig.jpg -set colorspace Gray -separate -average out.jpg
```

Convert JPEG format to PNG
```
convert in.jpg out.png
```