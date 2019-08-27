# -*- coding: utf-8 -*-
import os
import click
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import cv2

from sdcdup.utils import get_tile


def create_256_tiles(train_image_dir, train_tile_dir):
    n_tiles = 9

    os.makedirs(train_tile_dir, exist_ok=True)

    img_ids = os.listdir(train_image_dir)

    for img_id in tqdm(img_ids):
        img = None
        filebase, fileext = img_id.split('.')
        for idx in range(n_tiles):
            outfile = os.path.join(train_tile_dir, f'{filebase}_{idx}.{fileext}')
            if os.path.exists(outfile):
                continue
            if img is None:
                img = cv2.imread(os.path.join(train_image_dir, img_id))
            tile = get_tile(img, idx)
            cv2.imwrite(outfile, tile)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # TODO: 0 download dataset
    # TODO: 1 create_256_tiles
    # train_image_dir = os.path.join(project_root, os.getenv('RAW_DATA_DIR'), 'train_768')
    # train_tile_dir = os.path.join(project_root, os.getenv('PROCESSED_DATA_DIR'), 'train_256')
    train_image_dir = os.path.join(input_filepath, 'train_768')
    train_tile_dir = os.path.join(output_filepath, 'train_256')

    create_256_tiles(train_image_dir, train_tile_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_root = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
