# Results

The final task in this project is to join together all overlapping tiles into groups.
This [Notebook](notebooks/rebuild_overlap_groups.ipynb) contains the rebuilding code.

The overlap_groups are saved to rebuild_overlap_group.json.
For each group, the key is the name of the file I save the group image, and the values are themselves key value pairs that are the original tile filename (key) and offset from an origin common to all images in the group.
The offsets need to be multiplied by 256 to make sure the images overlap correctly.

Examples of some of the more interesting results are provided below.

![](notebooks/figures/results/overlap_109_13_19_0067a8e04.jpg)

![](notebooks/figures/results/overlap_116_21_28_02d772058.jpg)

![](notebooks/figures/results/overlap_198_34_30_02748d80d.jpg)

![](notebooks/figures/results/overlap_204_17_27_002943412.jpg)

![](notebooks/figures/results/overlap_254_29_31_0005d01c8.jpg)

![](notebooks/figures/results/overlap_291_35_29_005faf88a.jpg)

![](notebooks/figures/results/overlap_945_49_83_003d43308.jpg)

![](notebooks/figures/results/overlap_1426_70_123_0027854cc.jpg)

..._beautiful_