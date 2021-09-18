#!/usr/bin/env python3
"""
Executable script for running different experiments with our rule extraction
methods. Calls the backend code in dnn_rem and displays/serializes it as tables
for analysis.
"""

import argparse
import logging
import os
import sys
import time
import warnings
import yaml
import cProfile


from model_training import train_dnns
from experiment_runners.cross_validation import cross_validate_re
from experiment_runners.manager import ExperimentManager, EXPERIMENT_STAGES


################################################################################
## HELPER METHODS
################################################################################

def _to_val(x):
    try:
        return int(x)
    except ValueError:
        # Then this is not an int
        pass

    try:
        return float(x)
    except ValueError:
        # Then this is not an float
        pass

    if x.lower() in ["true"]:
        return True
    if x.lower() in ["false"]:
        return False

    return x


def build_parser():
    """
    Helper function to build our program's argument parser.

    :returns ArgumentParser: The parser for our program's configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Runs cross validation experiment with the given rule extraction '
            'method and our neural network training.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        default=None,
        help="initial configuration YAML file for our experiment's setup.",
        metavar="file.yaml",
    )
    parser.add_argument(
        '--n_folds',
        '-n',
        default=None,
        help='how many folds to use for our data partitioning.',
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--dataset_name',
        default=None,
        help="name of the dataset to be used for training.",
        metavar="name",
    )
    parser.add_argument(
        '--dataset_file',
        default=None,
        help="comma-separated-valued file containing our training data.",
        metavar="data.cvs",
    )
    parser.add_argument(
        '--rule_extractor',
        default=None,
        help=(
            "name of the extraction algorithm to be used to generate our "
            "rule set."
        ),
        metavar="name",
        choices=[
            "CART",
            "Clause-REM-D",
            "cREM-D",
            "DeepRED",
            "ECLAIRE",
            "eREM-D",
            "random_forest",
            "RandomForest",
            "REM-T",
            "sREM-D",
            'Pedagogical',
            'REM-D',
        ],
    )
    parser.add_argument(
        '--grid_search',
        action="store_true",
        default=None,
        help=(
            "whether we want to do a grid search over our model's "
            "hyperparameters. If the results of a previous grid search are "
            "found in the provided output directory, then we will use those "
            "rather than starting a grid search from scratch. This means that "
            "turning this flag on will overwrite any hyperparameters you "
            "provide as part of your configuration (if any)."
        ),
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        default=None,
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use the same directory as our dataset."
        ),
        metavar="path",

    )
    parser.add_argument(
        '--randomize',
        '-r',
        default=False,
        action="store_true",
        help=(
            "If set, then the random seeds used in our execution will not "
            "be fixed and the experiment will be randomized. By default, "
            "otherwise, all experiments are run with the same seed for "
            "reproducibility."
        ),

    )
    parser.add_argument(
        '--force_rerun',
        '-f',
        choices=(["all"] + EXPERIMENT_STAGES),
        default=None,
        help=(
            "If set and we are given as output directory the directory of a "
            "previous run, then we will overwrite any previous work starting "
            "from the provided stage (and all subsequent stages of the "
            "experiment) and redo all computations. Otherwise, we will "
            "attempt to use as much as we can from the previous run."
        ),
    )
    parser.add_argument(
        '--profile',
        help=(
            "prints out profiling statistics of the rule-extraction in terms "
            "of low-level calls used for the extraction method."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )

    return parser


################################################################################
## MAIN METHOD
################################################################################

def main():
    """
    Our main entry point method for our program's execution. Instantiates
    the argument parser and calls the appropriate methods using the given
    flags.
    """

    # First generate our argument parser
    parser = build_parser()
    args = parser.parse_args()

    # Now set up our logger
    logging.getLogger().setLevel(
        logging.DEBUG if args.debug else logging.INFO
    )
    logging.basicConfig(
        format='[%(levelname)s] %(message)s'
    )

    # Now see if a config file was passed or if all arguments come from the
    # command line
    if args.config is not None:
        # First deserialize and validate the given config file if it was
        # given
        if not os.path.exists(args.config):
            raise ValueError(
                f'Given config file "{args.config}" is not a valid path.'
            )
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {}
        if args.n_folds is None:
            # Then default it to 1
            # Here and below: the reason why we do not default it in the
            # argparser itself is because we want to have the option to
            # run between the mode where a config file has been provided but we
            # want to overwrite certain attributes of it and the case where no
            # config file was provided so we expect all arguments coming from
            # the command line.
            args.n_folds = 1
        if args.rule_extractor is None:
            # Default it to use our rule generation algorithm
            args.rule_extractor = "ECLAIRE"
        if None in [
            args.dataset_name,
            args.dataset_file,
        ]:
            # Then time to complain
            raise ValueError(
                'We expect to either be provided a valid configuration '
                'YAML file or to be explicitly provided arguments '
                'dataset_name and dataset_file. Otherwise, we do not have a '
                'complete parameterization for our experiment.'
            )

    # And we overwrite any arguments that were provided outside of our
    # config file:
    if args.n_folds is not None:
        config["n_folds"] = args.n_folds
    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    if args.rule_extractor is not None:
        config["rule_extractor"] = args.rule_extractor
    if args.dataset_file is not None:
        config["dataset_file"] = args.dataset_file
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.grid_search is not None:
        if "grid_search_params" not in config:
            config["grid_search_params"] = {}
        config["grid_search_params"]["enable"] = args.grid_search

    if args.randomize:
        # Then set the seed to be the current time
        config["random_seed"] = time.time()
    else:
        # Else let's fix it to a the answer to the universe
        config["random_seed"] = 42

    # And our initial stage to start overwriting, if any.
    start_rerun_stage = args.force_rerun or config.get("force_rerun")

    for param_path, value in args.param:
        var_names = list(map(lambda x: x.strip(), param_path.split(".")))
        current_obj = config
        for path_entry in var_names[:-1]:
            if path_entry not in config:
                current_obj[path_entry] = {}
            current_obj = current_obj[path_entry]
        current_obj[var_names[-1]] = _to_val(value)

    # Time to initialize our experiment manager
    with ExperimentManager(config, start_rerun_stage) as manager:
        # Generate our neural network, train it, and then extract the ruleset
        # that approximates it from it
        print(
            "Starting experiment with data being dumped at",
            manager.experiment_dir,
        )
        train_dnns.run(
            manager=manager,
            use_grid_search=manager.GRID_SEARCH_PARAMS.get("enable", False),
        )

        if args.profile:
            pr = cProfile.Profile()
            pr.enable()
        # Perform n fold cross validated rule extraction on the dataset
        cross_validate_re(manager=manager)
        # And turn off our profiler if we were using it
        if args.profile:
            pr.disable()
            pr.print_stats(sort='cumtime')

    # And that's all folks
    return 0

################################################################################
## ENTRY POINT
################################################################################

if __name__ == '__main__':
    # Make sure we do not print any useless warnings in our script to reduce
    # the amount of noise
    prev_tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            # execute code that will generate warnings
            sys.exit(main())
    finally:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = prev_tf_log_level

    # If we reached this point, then we are exiting with an error code
    sys.exit(1)
