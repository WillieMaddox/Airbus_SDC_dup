## Per Image Metrics

We only need to calculate the per-image metrics once when we first run the install script.
The installer will save the the results to intermediate files which can then be loaded much faster than having to recalculate the metrics again.  

Below we show a list of algorithms we tested to find duplicates.  

One option is to compare various per-image metrics using various image "similarity" algorithms:
- [image hashes](notebooks/eda/image_hashes.ipynb)
- [image entropy](notebooks/eda/image_entropy.ipynb)
- [image histograms](notebooks/eda/image_histograms.ipynb) 

Unfortunately, no single one of these nor any combination work particularly well across the entire dataset. 
They produce far too many false positives and false negatives to be useful.

### Image Hashes

* [md5 hash](https://docs.python.org/3/library/hashlib.html)
* [block-mean hash](https://www.phash.org/docs/pubs/thesis_zauner.pdf)
* [color-moment hash](http://www.naturalspublishing.com/files/published/54515x71g3omq1.pdf)

Use the `hashlib` python package to calculate md5 checksum.
Perceptual image hash functions are available through the contrib add-on package beginning with OpenCV 3.3.0.

### Image Entropy

* Cross Entropy?
* Shannon Entropy?

### grey-level co-occurrence matrix ([wiki](https://en.wikipedia.org/wiki/Co-occurrence_matrix))

* energy
* contrast
* homogeneity
* correlation

Available through [scikit-image](https://scikit-image.org/docs/stable/api/skimage.feature.html#greycomatrix)

See also, [Harris geospatial](https://www.harrisgeospatial.com/docs/backgroundtexturemetrics.html)

### Other useful per-image metrics.

* Is solid?
* Ship counts

## Overlap Metrics

* Binary pixel difference
* Absolute pixel difference

### Other Useful Sources

<https://en.wikipedia.org/wiki/Relative_change_and_difference>

<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.455.8550&rep=rep1&type=pdf>

<http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html>
