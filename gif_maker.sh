#!/bin/bash

# Author: Mateusz Janda (c) <mateusz janda at gmail com>
# Site: https://github.com/MateuszJanda/genetic-scii

if [[ -z $1 ]]; then
    echo "Warning - missing original file name."
    echo "Usage:"
    echo "   ./gif_maker.sh input.png"
else
    SCALE="416x416"
    echo "Be aware! Current scale:" $SCALE
    echo ""

    # Shrink original image
    orig_name=resize_$1
    convert $1 -resize $SCALE $orig_name

    # Shrink all snapshot_*.png files in current directory and merge with original file
    for i in snapshot_*.png;
    do
        # Resize image
        snap_name=resize_$i
        convert $i -resize $SCALE $snap_name

        # Merge original (resized) with snapshot (resized)
        merge_name=merge_$i
        convert $orig_name $snap_name +append $merge_name

        echo "Preparing $merge_name"
        # Cleanup
        rm $snap_name
    done

    # Removed resized original file
    rm $orig_name

    PALETTE_FILE="tmp_pallete.png"
    VIDEO_FILE="output_video.mp4"
    INPUT_FILES="merge_snapshot_%4d.png"
    OUTPUT_GIF="output.gif"
    FILTERS="fps=25"

    # Create video from *.png files
    ffmpeg -r 10 -i $INPUT_FILES -c:v libx264 -crf 0 -preset veryslow $VIDEO_FILE

    # Convert video file to .gif
    ffmpeg -v warning -i $VIDEO_FILE -vf "$FILTERS,palettegen" -y $PALETTE_FILE
    ffmpeg -v warning -i $VIDEO_FILE -i $PALETTE_FILE -lavfi "$FILTERS [x]; [x][1:v] paletteuse" -y $OUTPUT_GIF

    echo ""
    echo "End"
    echo "Check:" $VIDEO_FILE
    echo "Check:" $OUTPUT_GIF
fi
