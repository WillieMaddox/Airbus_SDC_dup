# -*- coding: utf-8 -*-
import os
import click
import logging
# from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from sdcdup.features import SDCImageContainer

allowed_score_types = ('bmh', 'cmh', 'enp', 'pix', 'px0')


@click.command()
@click.argument('score_types', type=str, nargs=-1)
def main(score_types):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    score_types = set([s.lower() for s in score_types])
    for score_type in score_types:
        assert score_type in allowed_score_types

    sdcic = SDCImageContainer()
    sdcic.load_image_overlap_properties(score_types=score_types)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_root = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
