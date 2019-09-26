# TODO

* Documentation
  -[ ] Add links to papers.
  -[ ] Add links to project files.
  -[x] Make new notebook showing differences in cropped images that result from jpeg compression.
* Organization and Refactoring
  -[ ] Move all plotting functions to visualize.py
  -[ ] Move preprocess_label_properties into preprocess_image_properties?
  -[ ] Use defaultdict instead of checking for "not in"
  -[ ] Don't use multiple folders for jupyter notebooks.  Just makes it hard to find.
  -[ ] Create a new SDCTile class that represents a single tile from SDCImage.
* Performance
  -[ ] Profile PyTorch prediction pipeline in PyCharm.
  -[ ] Parallelize the initialization in image_features
* Write unit tests
  -[ ] ...for most everything in utils.py
  -[ ] Use design pattern: Each test should have 3 parts: Arrange, Act, Assert
  -[ ] Verify rotate image {1, 2, 3} x 90 degrees in pytorch?
  -[ ] make sure `len(md5hashes) == len(set(md5hashes))`, otherwise we'll have to make our hashes longer.
* Net upgrades
  -[ ] Try using BatchNorm, or LayerNorm
  -[ ] Try an convolutional net. Remove max pooling layers and add stride 2 (or 3?) to convolutions
  -[ ] Try using inputs with shape (BATCH_SIZE, 2, 3, 256, 256) instead of (BATCH_SIZE, 6, 256, 256)?
* Importance sampling
  -[ ] Implement UCB like function?
  -[ ] Implement TF-IDF like function?
  -[ ] Use bmh hamming distance with as_scores=True as initial guess to weights?
* Use decision tree to show performance of DupNet over other metrics. (not sure how)
* Need examples of false positives and false negatives for each of the following.
  -[ ] image hashes
  -[ ] image entropy
  -[ ] image histograms
  -[ ] grey-level co-occurance matrix
  -[ ] ship overlap
* Dup Tool
  -[ ] Upgrade the dup_tool to use Bokeh or Plotly for interactions and rendering.

