
IFS=$(echo -en "\n\b")

# resize images
for f in $(find $1 -iname *.jpg | head -n $2); do echo $f; convert "$f" -resize 128x128 dataset/color/`basename "$f"`; done

# convert color images in dataset/color into gray images stored in dataset/gray
for f in $(ls --color=never dataset/color/*.JPG); do echo $f; convert "$f" -set colorspace Gray -separate -average dataset/gray/$(basename "$f"); done
