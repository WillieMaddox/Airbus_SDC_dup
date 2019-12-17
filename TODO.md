# TODO

* Documentation
  - [x] Add links to papers.
  - [x] Add links to project files.
  - [x] Make new notebook showing differences in cropped images that result from jpeg compression.
* Organization and Refactoring
  - [ ] Move all plotting functions to visualize.py
  - [x] ~~Move create_label_metrics into create_image_metrics?~~ Keep separate.
  - [x] Don't use multiple folders for jupyter notebooks.  Just makes it hard to find.
  - [x] ~~Create a new SDCTile class that represents a single tile from SDCImage.~~ Cancelled. Too slow.
* Performance
  - [x] Parallelize the creation of image_metrics.
  - [x] Parallelize the creation of overlap_scores.
* Write unit tests
  - [ ] ...for most everything in utils.py
  - [ ] Use design pattern: Each test should have 3 parts: Arrange, Act, Assert
  - [ ] Verify rotate image {1, 2, 3} x 90 degrees in pytorch?
  - [ ] make sure `len(md5hashes) == len(set(md5hashes))`, otherwise we'll have to make our hashes longer.
* Net upgrades
  - [ ] Try using BatchNorm, or LayerNorm
  - [ ] Try an all convolutional net. Remove max pooling layers and add stride 2 (or 3?) to convolutions
  - [ ] Try using inputs with shape (BATCH_SIZE, 2, 3, 256, 256) instead of (BATCH_SIZE, 6, 256, 256)?
* Importance sampling
  - [ ] Implement UCB like function?
  - [ ] Implement [TF-IDF](https://skymind.ai/wiki/bagofwords-tf-idf) like function?
  - [ ] Use bmh hamming distance with as_scores=True as initial guess to weights?
* Features
  - [x] Implement an efficient way to lower (or raise) the overlap matches threshold.
  - [x] Implement efficient way to remove truth from auto when they are added manually.
  - [x] Skip image pairs that have already been verified as duplicates.
  - [ ] Find out which remaining tile pairs have masks but aren't in dup_truth.
  - [ ] Provide examples of false positives and false negatives for each of the following:
    - [ ] image hashes
    - [ ] image entropy
    - [ ] image histograms
    - [ ] grey-level co-occurrence matrix
    - [ ] ship overlap
  - [ ] Use decision trees to show performance of DupNet over other metrics. (not sure how)
* Dup Tool
  - [ ] Upgrade the dup_tool to use Bokeh or Plotly for interactions and rendering.
  - [ ] Handle StopIteration exception on last record read.
