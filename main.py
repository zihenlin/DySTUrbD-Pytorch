"""
This model simulate the dynamics of a City under Covid-19 pandemic.
DySTUrbD-epi: Main

This program acts as an interface for model fusion,
or the project master file for debugging.
"""

import argparse

from model import DySTUrbD_Epi


def epi_run(args):
    """
    This function runs epidemiological model.
    """
    model = DySTUrbD_Epi(args)


if __name__ == "__main__":
    """
    Enable this program to run in standalone mode,
    while preserving the flexibility to be fused into other models.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticks", default=30, type=int, help="Max ticks for simulation."
    )
    parser.add_argument(
        "--out_dir", default="./outputs/", type=str, help="Directory for output files."
    )
    parser.add_argument(
        "--zones_dir",
        default="./data/dummy_zones.csv",
        type=str,
        help="Directory for zones data.",
    )
    parser.add_argument(
        "--roads_dir",
        default="./data/dummy_roads.csv",
        type=str,
        help="Directory for roads data.",
    )
    parser.add_argument(
        "--agents_dir",
        default="./data/civ_withCar_bldg_np.csv",
        type=str,
        help="Directory for agent-related data.",
    )
    parser.add_argument(
        "--buildings_dir",
        default="./data/bldg_with_inst_orig.csv",
        type=str,
        help="Directory for building-related data.",
    )
    parser.add_argument(
        "--infection_dir",
        default="./data/infection_prob.csv",
        type=str,
        help="Directory for infection probability data.",
    )
    parser.add_argument(
        "--admission_dir",
        default="./data/admission_prob.csv",
        type=str,
        help="Directory for admission probability data.",
    )
    parser.add_argument(
        "--mortality_dir",
        default="./data/mortality_prob.csv",
        type=str,
        help="Directory for mortality probability data.",
    )
    args = parser.parse_args()
    epi_run(args)
