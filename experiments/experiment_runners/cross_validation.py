from prettytable import PrettyTable
import logging
import numpy as np
import pandas as pd
import pickle
import sklearn
import tensorflow as tf

from model_training.train import load_model
from remix.evaluate_rules.evaluate import evaluate, evaluate_estimator
from remix.utils.resources import resource_compute
from remix.rules.ruleset import Ruleset


def _deserialize_rules(path):
    # Helper method to deserialize our rules and the resource
    # consumption it took.
    with open(path, 'rb') as f:
        return pickle.load(f)


def _serialize_rules(result, path):
    # Helper method to seserialize our rules and the resource
    # consumption it took into a given file
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    return result


def cross_validate_re(manager):
    # We will generate a pretty table for the end result so that it can
    # be pretty-printed at the end of the experiment and visually reported to
    # the user
    table = PrettyTable()
    regression = manager.DATASET_INFO.regression
    if regression:
        table.field_names = [
            "Fold",
            "NN Loss",
            f"{manager.RULE_EXTRACTOR.mode} Loss",
            f"{manager.RULE_EXTRACTOR.mode} MSE Fidelity",
            "Extraction Time (sec)",
            "Extraction Memory (MB)",
            "Ruleset Size",
            "Average Rule Length",
            "# of Terms",
        ]
    else:
        table.field_names = [
            "Fold",
            "NN Accuracy",
            'NN AUC',
            f"{manager.RULE_EXTRACTOR.mode} Accuracy",
            f"{manager.RULE_EXTRACTOR.mode} AUC",
            f"{manager.RULE_EXTRACTOR.mode} Fidelity",
            "Extraction Time (sec)",
            "Extraction Memory (MB)",
            "Ruleset Size",
            "Average Rule Length",
            "# of Terms",
        ]
    results_df = pd.DataFrame(data=[], columns=['fold'])

    # Extract rules from model from each fold
    table_rows = None
    for fold in range(1, manager.N_FOLDS + 1):

        ########################################################################
        ## Neural Network Evaluation
        ########################################################################

        # Get train and test data folds
        X_train, y_train, X_test, y_test = manager.get_fold_data(fold)

        # Path to neural network model for this fold
        model_file_path = manager.n_fold_model_fp(fold)
        nn_model = load_model(model_file_path)
        if regression:
            nn_loss = nn_model.evaluate(
                X_test,
                y_test,
                verbose=(
                    1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
                    else 0
                ),
            )
        else:
            nn_loss, _, nn_accuracy, majority_class = nn_model.evaluate(
                X_test,
                tf.keras.utils.to_categorical(
                    y_test,
                    num_classes=len(manager.DATASET_INFO.output_classes),
                ),
                verbose=(
                    1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
                    else 0
                ),
            )
            if len(manager.DATASET_INFO.output_classes) <= 2:
                nn_auc = sklearn.metrics.roc_auc_score(
                    y_test,
                    np.argmax(nn_model.predict(X_test), axis=-1),
                )
            else:
                nn_auc = 0

        ########################################################################
        ## Rule Extraction Evaluation
        ########################################################################

        # Path to extracted rules from that fold
        extracted_rules_file_path = manager.n_fold_rules_fp(fold)

        # Run our rule extraction only if it has not been done in the past
        # through a sequential checkpoint
        (surrogate, re_time, re_memory), _ = manager.serializable_stage(
            target_file=extracted_rules_file_path,
            execute_fn=lambda: resource_compute(
                function=manager.RULE_EXTRACTOR.run,
                model=nn_model,
                train_data=X_train,
                train_labels=y_train,
            ),
            serializing_fn=_serialize_rules,
            deserializing_fn=_deserialize_rules,
            stage_name="rule_extraction",
        )

        if isinstance(surrogate, Ruleset):
            # Serialize a human readable version of the rules always for
            # inspection
            with open(extracted_rules_file_path + ".txt", 'w') as f:
                for rule in surrogate:
                    f.write(str(rule) + "\n")
            # Now let's assign scores to our rules depending on what scoring
            # function we were asked to use for this experiment
            logging.debug(
                f'Evaluating rules extracted from '
                f'fold {fold}/{manager.N_FOLDS} ({surrogate.num_clauses()} '
                f'rules with {surrogate.num_terms()} different terms in them '
                f'on {len(y_test)} test points)...'
            )
            surrogate.rank_rules(
                X=X_train,
                y=y_train,
                score_mechanism=manager.RULE_SCORE_MECHANISM,
            )

            # Drop any rules if we are interested in dropping them
            surrogate.eliminate_rules(manager.RULE_DROP_PRECENT)

            # And actually evaluate them
            re_results = evaluate(
                ruleset=surrogate,
                X_test=X_test,
                y_test=y_test,
                high_fidelity_predictions=np.argmax(
                    nn_model.predict(X_test),
                    axis=1
                ) if (not regression) else nn_model.predict(X_test),
                num_workers=manager.EVALUATE_NUM_WORKERS,
                multi_class=(len(manager.DATASET_INFO.output_classes) > 2),
            )
        else:
            # Else we are talking about a decision tree classifier in here
            # And actually evaluate them
            re_results = evaluate_estimator(
                estimator=surrogate,
                X_test=X_test,
                y_test=y_test,
                high_fidelity_predictions=np.argmax(
                    nn_model.predict(X_test),
                    axis=1
                ) if (not regression) else nn_model.predict(X_test),
                regression=regression,
                multi_class=(len(manager.DATASET_INFO.output_classes) > 2),
            )

        ########################################################################
        ## Table writing and saving
        ########################################################################

        # Same some of this information into our dataframe
        results_df.loc[fold, 'fold'] = fold
        results_df.loc[fold, 'nn_loss'] = nn_loss
        if not regression:
            results_df.loc[fold, 'nn_accuracy'] = nn_accuracy
            results_df.loc[fold, 'nn_auc'] = nn_auc
            results_df.loc[fold, 'majority_class'] = majority_class
        results_df.loc[fold, 're_time (sec)'] = re_time
        results_df.loc[fold, 're_memory (MB)'] = re_memory
        if regression:
            results_df.loc[fold, 're_loss'] = re_results['loss']
            results_df.loc[fold, 're_mse_fid'] = re_results['mse_fid']
        else:
            results_df.loc[fold, 're_acc'] = re_results['acc']
            results_df.loc[fold, 're_fid'] = re_results['fid']
            results_df.loc[fold, 're_auc'] = re_results['auc']
            results_df.loc[fold, 'output_classes'] = str(
                re_results['output_classes']
            )
            results_df.loc[fold, 're_n_rules_per_class'] = str(
                re_results.get('n_rules_per_class', 0)
            )
        results_df.loc[fold, 'min_n_terms'] = str(
            re_results.get('min_n_terms', 0)
        )
        results_df.loc[fold, 'max_n_terms'] = str(
            re_results.get('max_n_terms', 0)
        )
        results_df.loc[fold, 'av_n_terms_per_rule'] = str(
            re_results.get('av_n_terms_per_rule', 0)
        )
        results_df.loc[fold, 're_terms'] = re_results.get(
            'n_unique_terms',
            0
        )

        if regression:
            logging.debug(
                f"Rule extraction for fold {fold} took a total of "
                f"{re_time} sec "
                f"and {re_memory} MB to obtain "
                f"testing loss of {re_results['loss']} compared to the "
                f"loss of the neural network {nn_loss}."
            )
        else:
            logging.debug(
                f"Rule extraction for fold {fold} took a total of "
                f"{re_time} sec "
                f"and {re_memory} MB to obtain "
                f"testing accuracy {re_results['acc']} compared to the "
                f"accuracy of the neural network {nn_accuracy}."
            )

        # Fill up our pretty table
        if isinstance(surrogate, Ruleset):
            avg_rule_length = np.array(re_results['av_n_terms_per_rule'])
            avg_rule_length *= np.array(re_results['n_rules_per_class'])
            avg_rule_length = sum(avg_rule_length)
            avg_rule_length /= sum(re_results['n_rules_per_class'])
            avg_rule_length = round(avg_rule_length, manager.ROUNDING_DECIMALS)
            num_rules = sum(re_results['n_rules_per_class'])
        else:
            num_rules = 0
            avg_rule_length = 0
        if regression:
            new_row = [
                round(nn_loss, manager.ROUNDING_DECIMALS),
                round(re_results['loss'], manager.ROUNDING_DECIMALS),
                round(re_results['mse_fid'], manager.ROUNDING_DECIMALS),
                round(re_time,  manager.ROUNDING_DECIMALS),
                round(re_memory, manager.ROUNDING_DECIMALS),
                num_rules,
                avg_rule_length,
                re_results.get('n_unique_terms', 0),
            ]
        else:
            new_row = [
                round(nn_accuracy, manager.ROUNDING_DECIMALS),
                round(nn_auc, manager.ROUNDING_DECIMALS),
                round(re_results['acc'], manager.ROUNDING_DECIMALS),
                round(re_results['auc'], manager.ROUNDING_DECIMALS),
                round(re_results['fid'], manager.ROUNDING_DECIMALS),
                round(re_time,  manager.ROUNDING_DECIMALS),
                round(re_memory, manager.ROUNDING_DECIMALS),
                num_rules,
                avg_rule_length,
                re_results.get('n_unique_terms', 0),
            ]
        if table_rows is None:
            table_rows = np.expand_dims(np.array(new_row), axis=0)
        else:
            table_rows = np.concatenate(
                [table_rows, np.expand_dims(new_row, axis=0)],
                axis=0,
            )
        table.add_row([fold] + new_row)

        # Finally, log this int the progress bar if not on quiet mode to get
        # some feedback
        if regression:
            logging.info(
                f'Rule set test loss for fold {fold}/{manager.N_FOLDS} '
                f'is {round(re_results["loss"], 3)}, MSE fidelity is '
                f'{round(re_results["mse_fid"], 3)}, and size of rule set '
                f'is {num_rules}'
            )
        else:
            logging.info(
                f'Rule set test accuracy for fold {fold}/{manager.N_FOLDS} '
                f'is {round(re_results["acc"], 3)}, AUC is '
                f'{round(re_results["auc"], 3)}, fidelity is '
                f'{round(re_results["fid"], 3)}, and size of rule set '
                f'is {num_rules}'
            )

    # Now that we are done, let's serialize our dataframe for further analysis
    results_df.to_csv(manager.N_FOLD_RESULTS_FP, index=False)

    # Finally, let's include an average column that also has a standard
    # deviation included in it
    avgs = list(map(
        lambda x: round(x,  manager.ROUNDING_DECIMALS),
        np.mean(table_rows, axis=0)
    ))
    stds = list(map(
        lambda x: round(x,  manager.ROUNDING_DECIMALS),
        np.std(table_rows, axis=0)
    ))
    avgs = [
        f'{avg} ± {std}' for (avg, std) in zip(avgs, stds)
    ]
    table.add_row(
        ["avg"] +
        avgs
    )

    # And display our results as a pretty table for the user to inspect quickly
    if logging.getLogger().getEffectiveLevel() not in [
        logging.WARNING,
        logging.ERROR,
    ]:
        print(table)
    # And always serialize our table results into a pretty-printed txt file
    with open(manager.SUMMARY_FILE, 'w') as f:
        f.write(table.get_string())
