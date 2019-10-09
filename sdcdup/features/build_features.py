# -*- coding: utf-8 -*-
import os
import click
import logging
# from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from sdcdup.features import create_image_overlap_properties

allowed_n_matching_tiles = (9, 6, 4, 3, 2, 1)
allowed_score_types = ('bmh', 'cmh', 'enp', 'pix', 'px0')


@click.command()
@click.argument('n_matching_tiles_str', type=str, nargs=1)
@click.argument('score_types', type=str, nargs=-1)
def main(n_matching_tiles_str, score_types):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    n_matching_tiles_list = set(map(int, list(n_matching_tiles_str)))
    for n_matching_tiles in n_matching_tiles_list:
        assert n_matching_tiles in allowed_n_matching_tiles

    score_types = set([s.lower() for s in score_types])
    for score_type in score_types:
        assert score_type in allowed_score_types

    create_image_overlap_properties(n_matching_tiles_list, score_types=score_types)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_root = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
