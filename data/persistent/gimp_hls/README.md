## HSL modified image files created with GIMP

The images in these directories, were used to derive the function `channel_shift` in [utils.py](../../../sdcdup/utils.py)

The filename format is as follows:
1. Channel: `H`ue, `S`aturation, or `L`ightness
2. Sign: `p`lus or `m`inus
3. Gain: `(000, 180]` degrees for Hue and `(000, 100]` percent for Saturation and Lightness.

The names of these directories correspond to the basename of the original image used to generate the perturbed files.

For example, `./03a5fd8d2/Sp025.jpg` is the result of increasing (i.e. `p`) the `S`aturation of `03a5fd8d2.jpg` by 25 percent.
`./03a5fd8d2/Hm135.jpg` is the result of decreasing (i.e. `m`) the `H`ue of `03a5fd8d2.jpg` by 135 degrees.
Note that `Hm180` is the same as `Hp180`.

![hls_03](../../../notebooks/figures/03a5fd8d2.jpg_676f4cfd0.jpg_08.jpg)

(See [hls_shift.ipynb](../../../notebooks/eda/hls_shift.ipynb) for further analysis.)