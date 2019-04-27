#!/bin/bash

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.th.300.vec.gz
gunzip -d cc.th.300.vec.gz
rm cc.th.300.vec.gz
