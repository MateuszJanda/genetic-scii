## Dependencies
Tested with:
- Pillow           6.2.0


## ImageMagic:

Reduced just enough so as to best fit into the given size
convert orig.jpg -resize 64x64 shrink.jpg

Fore new size (ignore aspect ratio)
convert orig.jpg -resize 400x400\! shrink.jpg

Change RGB to grayscale
convert orig.jpg -set colorspace Gray -separate -average out.jpg

Convert JPEG format to PNG
convert in.jpg out.png