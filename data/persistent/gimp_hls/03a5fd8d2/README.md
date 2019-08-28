## gimp-altered image files.

These images were used to derive the function `channel_shift` in [utils.py](../../../../sdcdup/utils.py)  (See [hls_shift.ipynb](../../../../notebooks/eda/hls_shift.ipynb) for the analysis.)

The filename format is as follows:
1. Channel: `H`ue, `S`aturation, or `L`ightness
2. Sign: `p`lus or `m`inus
3. Gain: `(000, 180]` degrees for Hue and `(000, 100]` percent for Saturation and Lightness.

The name of this directory corresponds to the basename of the original image used to generate these files.

For example, `Sp025.jpg` is the result of increasing (i.e. `p`) the `S`aturation of `03a5fd8d2.jpg` by 25 percent,
and `Hm135.jpg` is the result of decreasing (i.e. `m`) the `H`ue of `03a5fd8d2.jpg` by 135 degrees.
Note that `Hm180` is the same as `Hp180`.

