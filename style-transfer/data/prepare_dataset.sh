#!/bin/sh
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
mv train2014 coco
rm -rf train2014.zip
