# coding=utf-8

import os
import itertools

import numpy as np
import cv2
from osgeo import gdal, gdal_array

from tqdm import tqdm
from parse import parse

from sdcdup.utils import ij_pairs_3x3
from sdcdup.utils import ijpair2idx
from sdcdup.utils import overlap_tag_pairs
from sdcdup.utils import generate_tag_pair_lookup
from sdcdup.utils import read_duplicate_truth
from sdcdup.utils import write_duplicate_truth

R = 6378137.0  # Radius of the earth in m


def ll_to_m(lat1, lon1, lat2, lon2):
    """
    returns the distance between two points
    along the geodesic of the "great sphere"
    return value is always positive
    """
    dlat = np.deg2rad(lat2)-np.deg2rad(lat1)
    dlon = np.deg2rad(lon2)-np.deg2rad(lon1)
    cos_lat1 = np.cos(np.deg2rad(lat1))
    cos_lat2 = np.cos(np.deg2rad(lat2))
    sin_dlat2 = np.sin(dlat / 2.0)
    sin_dlon2 = np.sin(dlon / 2.0)
    a = cos_lat1 * cos_lat2 * sin_dlon2 * sin_dlon2 + sin_dlat2 * sin_dlat2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def create_blocks_list(crop_region, block_shape):
    """Cretes a list of block reading coordinates.

    :param crop_region: Offsets and shape of the region of interest.
       (xoff, yoff, xsize, ysize)
    :param block_shape: Width and height of each block.
    """
    image_cols = crop_region[2]
    image_rows = crop_region[3]
    block_width = block_shape[0]
    block_height = block_shape[1]
    # Get the number of blocks.
    x_blocks = int((image_cols + block_width - 1) / block_width)
    y_blocks = int((image_rows + block_height - 1) / block_height)

    blocks = []
    for block_row in range(0, y_blocks):
        if block_row == y_blocks - 1:
            valid_y = image_rows - block_row * block_height
        else:
            valid_y = block_height
        yoff = block_row * block_height + crop_region[1]
        for block_col in range(0, x_blocks):
            if block_col == x_blocks - 1:
                valid_x = image_cols - block_col * block_width
            else:
                valid_x = block_width
            xoff = block_col * block_width + crop_region[0]
            blocks.append((xoff, yoff, valid_x, valid_y))
    return blocks


def scale_uint16_to_uint8(array, img_range=None):
    """Projects a range of values into a grayscale image.

    :param array: A Numpy array containing the image data.
    :param img_range: specified range of values or None to use
    the range of the image (minimum and maximum).
    """
    if img_range:
        mn = img_range[0]
        mx = img_range[1]
    else:
        mn = array.min()
        mx = array.max()

    if mx == mn:
        factor = mx / 65536
    else:
        interval = mx - mn
        factor = 256.0 / interval

    output = array * factor
    return output.astype(np.uint8)


def display_image(img_id, image):
    """displays an image file."""
    cv2.imshow(img_id, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_filename_generator(row, col):
    return f"r{row:03d}_c{col:03d}.jpg"


class FileInfo:
    """A class holding information about a GDAL file."""
    def __init__(self, filename, base_dir):
        self.filename = filename
        self.file = os.path.basename(filename)
        self.handle = filename.split(base_dir)[-1].rsplit(".")[0]
        self.output_dir = os.path.join("output/datasets", self.handle)
        self.raster_xsize = None
        self.raster_ysize = None
        self.band_colors = []
        self.data_type = None
        self.block_size = None
        self.no_data_value = None
        self.projection = None
        self.gt = None
        self._gsd = None
        self.image_colors = ("Red", "Green", "Blue")
        self.draw_n_times = 0
        self.tile_codes = None
        self.tile_size = 256
        self.mosaic_size = 3
        self.image_size = self.tile_size * self.mosaic_size
        self.black_cutoff = 10
        self.white_cutoff = 255 - self.black_cutoff

    def process(self):
        """
        Initialize file_info from filename
        filename -- Name of file to read.
        Returns True on success or False if the file can't be opened.
        """
        ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
        if ds is None:
            return False

        self.raster_xsize = ds.RasterXSize
        self.raster_ysize = ds.RasterYSize
        self.projection = ds.GetProjection()
        self.gt = ds.GetGeoTransform()

        self.data_type = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        self.block_size = ds.GetRasterBand(1).GetBlockSize()
        self.no_data_value = ds.GetRasterBand(1).GetNoDataValue()

        for i in range(ds.RasterCount):
            band = ds.GetRasterBand(i + 1)
            assert self.data_type == gdal.GetDataTypeName(band.DataType)
            assert self.block_size == band.GetBlockSize()
            assert self.no_data_value == band.GetNoDataValue()
            self.band_colors.append(gdal.GetColorInterpretationName(band.GetColorInterpretation()))

        # cleanup
        ds = None
        return True

    @property
    def gsd(self):
        # ground sample distance
        # return value should be in units of meters.
        if self._gsd is None:
            if abs(self.gt[0]) < 180 and abs(self.gt[3]) < 180 and abs(self.raster_xsize * self.gt[1]) < 180 and abs(self.raster_ysize * self.gt[5]) < 180:
                self._gsd = self.gt[1] * 111412.2402  # meters of 1 degree latitude at the equator.
            else:
                self._gsd = self.gt[1]
        return self._gsd

    def print_summary(self):
        print(f'bands {len(self.band_colors)}', end=' ')
        print(f'({self.raster_xsize:>5},{self.raster_ysize:>5})', end=' ')
        print(f'{self.data_type:>6}', end=' ')
        print(f'[{self.block_size[0]:>5},{self.block_size[1]:>5}]', end=' ')
        print(f'{self.gsd:>6.1f}', end=' ')
        print(self.handle)

    def generate_images_and_tiles(self, colors=("Red", "Green", "Blue")):

        self.print_summary()

        self.image_colors = colors or (None, None, None)
        color_indexes = []
        for c in self.image_colors:
            for i, band_color in enumerate(self.band_colors):
                if band_color == c:
                    color_indexes.append(i)
                    break
            else:
                if c is None:
                    color_indexes.append(None)
                else:
                    raise ValueError

        # If we still have Nones, fill each with random channel.
        if None in color_indexes:
            choices = []
            n = color_indexes.count(None)
            for i in range(len(self.band_colors)):
                if i not in color_indexes:
                    choices.append(i)
            assert len(choices) >= n
            for choice in np.random.choice(choices, n, replace=False):
                color_indexes[color_indexes.index(None)] = choice

        color_indexes = np.array(color_indexes)
        ds = gdal.Open(self.filename, gdal.GA_ReadOnly)

        extra_cols = self.raster_xsize % self.tile_size
        extra_rows = self.raster_ysize % self.tile_size
        cols = self.raster_xsize - extra_cols
        rows = self.raster_ysize - extra_rows
        xoff = extra_cols // 2
        yoff = extra_rows // 2
        blocks_list = create_blocks_list((xoff, yoff, cols, rows), (self.tile_size, self.tile_size))

        # TODO: Verify why they calculate x_block and y_blocks this way.
        x_blocks = int((cols + self.tile_size - 1) / self.tile_size)
        y_blocks = int((rows + self.tile_size - 1) / self.tile_size)
        numeric_type = gdal_array.GDALTypeCodeToNumericTypeCode(gdal.GetDataTypeByName(self.data_type))
        tile_grid = np.zeros((y_blocks, x_blocks, self.tile_size, self.tile_size, 3), dtype=numeric_type)
        for index, block in enumerate(blocks_list):
            # print(f"{index % x_blocks:>3}", end=' ')
            # print(f"{index // x_blocks:>3}", end=' ')
            # print(f"{block[0]:>6}", end=' ')
            # print(f"{block[1]:>6}")
            block_data = ds.ReadAsArray(*block)
            block_data = block_data[color_indexes]
            block_data = block_data.transpose(1, 2, 0)  # CHW -> HWC
            tile_grid[index // x_blocks, index % x_blocks] = block_data

        if self.data_type == "UInt16":
            tile_grid = scale_uint16_to_uint8(tile_grid)

        self.generate_tiles(tile_grid)
        self.generate_images(tile_grid)

        # cleanup
        ds = None

    def generate_tiles(self, tile_grid):

        y_blocks, x_blocks, h, w, c = tile_grid.shape
        assert h == self.tile_size
        output_dir = os.path.join(self.output_dir, f"images_{self.tile_size}")
        os.makedirs(output_dir, exist_ok=True)
        display_probability = self.draw_n_times / (y_blocks * x_blocks)
        self.tile_codes = np.zeros((y_blocks, x_blocks), dtype=np.int8)

        for y_block in range(y_blocks):
            for x_block in range(x_blocks):

                tile = tile_grid[y_block, x_block]
                tile_id = image_filename_generator(y_block, x_block)
                tile_filename = os.path.join(output_dir, tile_id)

                if np.all(np.max(tile, axis=(0, 1)) < self.black_cutoff):  # ignore all nearly black tiles
                    self.tile_codes[y_block, x_block] = -1
                elif np.all(np.min(tile, axis=(0, 1)) > self.white_cutoff):  # ignore all nearly white tiles
                    self.tile_codes[y_block, x_block] = 1
                else:
                    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    if not os.path.exists(tile_filename):
                        cv2.imwrite(tile_filename, tile)
                    if np.random.random() < display_probability:
                        display_image(tile_id, tile)

        self.flag_black_and_white(self.tile_codes, self.tile_size, "tiles")

    def generate_images(self, tile_grid):

        y_blocks, x_blocks, h, w, c = tile_grid.shape
        x_blocks -= (self.mosaic_size - 1)
        y_blocks -= (self.mosaic_size - 1)
        assert h * self.mosaic_size == self.image_size
        output_dir = os.path.join(self.output_dir, f"images_{self.image_size}")
        os.makedirs(output_dir, exist_ok=True)
        display_probability = self.draw_n_times / (y_blocks * x_blocks)

        image_codes = np.zeros((y_blocks, x_blocks), dtype=np.int8)
        image = np.zeros((self.image_size, self.image_size, c), dtype=np.uint8)

        for y_block in range(y_blocks):
            for x_block in range(x_blocks):

                image[0 * h:1 * h, 0 * w:1 * w] = tile_grid[y_block + 0, x_block + 0]
                image[0 * h:1 * h, 1 * w:2 * w] = tile_grid[y_block + 0, x_block + 1]
                image[0 * h:1 * h, 2 * w:3 * w] = tile_grid[y_block + 0, x_block + 2]
                image[1 * h:2 * h, 0 * w:1 * w] = tile_grid[y_block + 1, x_block + 0]
                image[1 * h:2 * h, 1 * w:2 * w] = tile_grid[y_block + 1, x_block + 1]
                image[1 * h:2 * h, 2 * w:3 * w] = tile_grid[y_block + 1, x_block + 2]
                image[2 * h:3 * h, 0 * w:1 * w] = tile_grid[y_block + 2, x_block + 0]
                image[2 * h:3 * h, 1 * w:2 * w] = tile_grid[y_block + 2, x_block + 1]
                image[2 * h:3 * h, 2 * w:3 * w] = tile_grid[y_block + 2, x_block + 2]
                image_id = image_filename_generator(y_block, x_block)
                image_filename = os.path.join(output_dir, image_id)

                if image.max() < self.black_cutoff:  # ignore all nearly black images
                    image_codes[y_block, x_block] = -1
                elif image.min() > self.white_cutoff:  # ignore all nearly white images
                    image_codes[y_block, x_block] = 1
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if not os.path.exists(image_filename):
                        cv2.imwrite(image_filename, image)
                    if np.random.random() < display_probability:
                        display_image(image_id, image)

        self.flag_black_and_white(image_codes, self.image_size, "images")
        self.generate_duplicate_truth(x_blocks, y_blocks, image_codes)

    def flag_black_and_white(self, code_array, size, result_type_str="results"):
        # TODO This is temporary.  Need to be able to flag tiles for ANY solid color.
        black = np.where(code_array < 0)
        if len(black[0]) > 0:
            print(f"found {len(black[0]):>3} black {result_type_str}.")
            np.save(os.path.join(self.output_dir, f"black_{str(size)}.npy"), black)

        white = np.where(code_array > 0)
        if len(white[0]) > 0:
            print(f"found {len(white[0]):>3} white {result_type_str}.")
            np.save(os.path.join(self.output_dir, f"white_{str(size)}.npy"), white)

    def generate_duplicate_truth(self, x_blocks, y_blocks, image_codes):
        """
        1 2 3 2 1     (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2)
        2 4 6 4 2     (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2)
        3 6 T 6 3 --> ( 0,-2), ( 0,-1),         ( 0,1), ( 0,2)
        2 4 6 4 2     ( 1,-2), ( 1,-1), ( 1,0), ( 1,1), ( 1,2)
        1 2 3 2 1     ( 2,-2), ( 2,-1), ( 2,0), ( 2,1), ( 2,2)

        """
        img2_offsets = [(-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
                        (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
                        ( 0,-2), ( 0,-1),         ( 0,1), ( 0,2),
                        ( 1,-2), ( 1,-1), ( 1,0), ( 1,1), ( 1,2),
                        ( 2,-2), ( 2,-1), ( 2,0), ( 2,1), ( 2,2)]

        img1_overlap_tags = ['0000', '0001', '0002', '0102', '0202',
                             '0010', '0011', '0012', '0112', '0212',
                             '0020', '0021',         '0122', '0222',
                             '1020', '1021', '1022', '1122', '1222',
                             '2020', '2021', '2022', '2122', '2222']

        duplicate_truth = {}
        for img1_y_block in range(y_blocks):
            for img1_x_block in range(x_blocks):

                if image_codes[img1_y_block, img1_x_block] != 0:
                    continue
                img1_id = image_filename_generator(img1_y_block, img1_x_block)

                for img1_overlap_tag, (y_off, x_off) in zip(img1_overlap_tags, img2_offsets):

                    img2_x_block = img1_x_block + x_off
                    img2_y_block = img1_y_block + y_off
                    if img2_x_block < 0 or img2_y_block < 0:
                        continue
                    if img2_x_block >= x_blocks or img2_y_block >= y_blocks:
                        continue
                    if image_codes[img2_y_block, img2_x_block] != 0:
                        continue
                    img2_id = image_filename_generator(img2_y_block, img2_x_block)
                    if img1_id < img2_id:
                        key = (img1_id, img2_id, img1_overlap_tag)
                    else:
                        key = (img2_id, img1_id, overlap_tag_pairs[img1_overlap_tag])
                    if key in duplicate_truth:
                        continue

                    duplicate_truth[key] = 1

        duplicate_truth_file = os.path.join(self.output_dir, "duplicate_truth.txt")
        write_duplicate_truth(duplicate_truth_file, duplicate_truth)

    def write_acquisition_image(self):
        pass


def write_duplicate_truth_paths(duplicate_truth_paths, rootpath=None, filename="duplicate_truth_paths.txt"):

    filename = os.path.join(rootpath, filename) if rootpath else filename
    with open(filename, 'w') as ofs:
        for duplicate_truth_path in sorted(duplicate_truth_paths):
            ofs.write(duplicate_truth_path + '\n')


def load_duplicate_truth_paths(rootpath=None, filename="duplicate_truth_paths.txt"):

    filename = os.path.join(rootpath, filename) if rootpath else filename
    with open(filename, 'r') as ifs:
        duplicate_truth_paths = ifs.read().strip().split()
    return duplicate_truth_paths


def create_dataset_from_truth(rootpath=None, filename="duplicate_truth_paths.txt"):

    tpl = generate_tag_pair_lookup()

    dup_truth_paths = load_duplicate_truth_paths(rootpath=rootpath, filename=filename)
    n_dup_image_pairs = 0
    img_ids = set()
    img_overlap_pairs = {}
    black_tiles_dict = {}
    white_tiles_dict = {}

    for dup_truth_path in tqdm(dup_truth_paths):

        dup_truth_path = os.path.join(rootpath, dup_truth_path) if rootpath else dup_truth_path
        dup_truth_filename = os.path.join(dup_truth_path, "duplicate_truth.txt")
        dup_truth = read_duplicate_truth(dup_truth_filename)

        check_black = False
        black_tiles_filename = os.path.join(dup_truth_path, "black_256.npy")
        if os.path.exists(black_tiles_filename):
            check_black = True
            black_tiles = np.load(black_tiles_filename)
            black_tiles_dict[dup_truth_path] = set((t[0], t[1]) for t in zip(*black_tiles))

        check_white = False
        white_tiles_filename = os.path.join(dup_truth_path, "white_256.npy")
        if os.path.exists(white_tiles_filename):
            check_white = True
            white_tiles = np.load(white_tiles_filename)
            white_tiles_dict[dup_truth_path] = set((t[0], t[1]) for t in zip(*white_tiles))

        img_filepath = os.path.join(dup_truth_path, 'images_768')
        # Collect all image pairs flagged as duplicates.
        # But exclude solid color tiles of equal color.

        for (img1_id, img2_id, img1_overlap_tag), is_dup in dup_truth.items():

            if not is_dup:
                print(img1_id, img2_id, img1_overlap_tag)
                raise ValueError("non-dup found!!! This script isn't set up to handle verified non-dup truth yet.")

            row1, col1 = parse('r{:3d}_c{:3d}.jpg', img1_id)
            row2, col2 = parse('r{:3d}_c{:3d}.jpg', img2_id)
            img1_id = os.path.join(img_filepath, img1_id)
            img2_id = os.path.join(img_filepath, img2_id)

            for idx1, idx2 in tpl[img1_overlap_tag]:
                if check_black:
                    i, j = ij_pairs_3x3[idx1]
                    if (row1 + i, col1 + j) in black_tiles_dict[dup_truth_path]:
                        continue
                    k, l = ij_pairs_3x3[idx2]
                    if (row2 + k, col2 + l) in black_tiles_dict[dup_truth_path]:
                        continue
                if check_white:
                    i, j = ij_pairs_3x3[idx1]
                    if (row1 + i, col1 + j) in white_tiles_dict[dup_truth_path]:
                        continue
                    k, l = ij_pairs_3x3[idx2]
                    if (row2 + k, col2 + l) in white_tiles_dict[dup_truth_path]:
                        continue
                img_overlap_pairs[(img1_id, img2_id, idx1, idx2)] = is_dup

            img_ids.add(img1_id)
            img_ids.add(img2_id)
            n_dup_image_pairs += 1

    n_dup_tile_pairs = len(img_overlap_pairs)
    print(f"Number of unique images: {len(img_ids)}")
    print(f"Number of duplicate image pairs: {n_dup_image_pairs}")
    print(f"Number of non-dup/dup tiles: {0:>8}/{n_dup_tile_pairs}")

    for img1_id in tqdm(img_ids):  # loop through all the 768's
        dup_truth_path, img_filename = img1_id.rsplit("/images_768/")
        row, col = parse('r{:3d}_c{:3d}.jpg', img_filename)
        ij_pairs = list(ij_pairs_3x3)

        if dup_truth_path in black_tiles_dict:
            for i, j in ij_pairs_3x3:
                if (row+i, col+j) in black_tiles_dict[dup_truth_path]:
                    ij_pairs.remove((i, j))

        if dup_truth_path in white_tiles_dict:
            for i, j in ij_pairs_3x3:
                if (row+i, col+j) in white_tiles_dict[dup_truth_path]:
                    ij_pairs.remove((i, j))

        comb_iter = itertools.combinations(ij_pairs, 2)
        for ij1, ij2 in comb_iter:
            idx1 = ijpair2idx[ij1]
            idx2 = ijpair2idx[ij2]
            overlap_key = (img1_id, img1_id, idx1, idx2)
            if overlap_key in img_overlap_pairs:
                print(overlap_key)
                continue
            img_overlap_pairs[overlap_key] = 0

        if len(img_overlap_pairs) > 2 * n_dup_tile_pairs:
            break

    print(f"Number of non-dup/dup tiles: {len(img_overlap_pairs) - n_dup_tile_pairs:>8}/{n_dup_tile_pairs}")

    # img_overlap_pairs = dict(**dup_overlap_tiles, **nondup_overlap_tiles)
    return [(img1_id, img2_id, idx1, idx2, is_dup) for (img1_id, img2_id, idx1, idx2), is_dup in img_overlap_pairs.items()]


if __name__ == '__main__':

    nas_root = "/mnt/nfs/DATA/geos/"
    borg_root = "/media/Borg_LS/DATA/geos/"

    # acquisition_files = [
    #     ("high-res-ports", borg_root + "Skysat/high-res-ports/20160928_020257_SSC1d1_0015_L3A_visual.tif"),
    #     ("rome-italy", borg_root + "Skysat/rome-italy/20180828_095115_ssc3_u0001_visual.tif"),
    #     ("chicago", borg_root + "Inria/chicago/chicago20.tif"),
    #     ("Utah", borg_root + "cowc/Utah/12TVL160160.tif"),
    #     ("naip", nas_root + "naip/34086/m_3408662_ne_16_1_20150726.tif"),
    #     ("TCA", nas_root + "TCA_LiDAR/Naco-Orthos/az12-1213.tif")
    # ]
    # for handle, filename in acquisition_files:
    #     file_info = FileInfo(filename)
    #     if file_info.process():
    #         file_info.generate_images_and_tiles(handle=handle)

    duplicate_truth_paths = []
    dataset_names = ['cowc', 'Inria', 'kaggle', 'Skysat']
    for dataset_name in dataset_names:
        base_dir = os.path.join(borg_root, dataset_name)
        for dirpath, dirnames, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename[-4:] not in (".tif", ".TIF"):
                    continue
                if filename[:-4].endswith('pansharpened'):
                    continue
                if filename[:-4].endswith('pansharp'):
                    continue
                if filename[:-4].endswith('analytic'):
                    continue
                if filename[:-4].endswith('_dn'):
                    continue
                file_info = FileInfo(os.path.join(dirpath, filename), borg_root)
                if not file_info.process():
                    continue
                # if file_info.handle.startswith("Inria"):
                #     continue
                if 'Red' not in file_info.band_colors:
                    continue
                if len(file_info.band_colors) < 3:
                    continue
                if file_info.raster_xsize < 4*256:
                    continue
                if file_info.raster_ysize < 4*256:
                    continue
                file_info.generate_images_and_tiles()
                duplicate_truth_paths.append(file_info.handle)

    print(f"Number of duplicate truth files: {len(duplicate_truth_paths)}")
    write_duplicate_truth_paths(duplicate_truth_paths, "output/datasets")

    create_dataset_from_truth("output/datasets")
