"""
Configurations of the different supported datasets for training.
"""

import numpy as np
import logging
import pandas as pd

from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler

from data.descriptors import (
    FeatureDescriptor, RealDescriptor, DiscreteNumericDescriptor,
    TrivialCatDescriptor, OutputClass, DatasetDescriptor
)


################################################################################
## Global Variables
################################################################################

# The names of all the datasets whose configurations we currently support
AVAILABLE_DATASETS = [
    'Artif-1',
    'Artif-2',
    'BreastCancer',
    'ForestCoverType',
    'GlassIdentification',
    'Iris',
    'LetterRecognition',
    'LetterRecognitionComplete',
    'MAGIC',
    'MB-1004-GE-2Hist',
    'MB-Clin-DR',
    'MB-Clin-ER',
    'MB-ClinP-ER',
    'MB-GE-2Hist',
    'MB-GE-6Hist',
    'MB-GE-Clin-ER',
    'MB-GE-ClinP-ER',
    'MB-GE-DR',
    'MB-GE-ER',
    'MB-ImageVec5-6Hist',
    'MB-ImageVec50-6Hist',
    'MB_GE_CDH1_2Hist',
    'mb_imagevec50_2Hist',
    'mb_imagevec50_DR',
    'mb_imagevec50_ER',
    'MiniBooNE',
    'MNIST',
    'MNIST-Complete',
    'PARTNER-Clinical',
    'PARTNER-Genomic',
    'SARCOS',
    'TCGA-PANCAN',
    'WineQuality',
    'WineQualityClassification',
    'XOR',
]


################################################################################
## Helper Methods
################################################################################

def unit_scale_preprocess(X_train, y_train, X_test=None, y_test=None):
    """
    Simple scaling preprocessing function to scale the X matrix so that all of
    it's features are in [0, 1]

    :param np.array X_train: 2D matrix of data points to be used for training.
    :param np.array y_train: 1D matrix of labels for the given training data
        points.
    :param np.array X_test: optional 2D matrix of data points to be used for
        testing.
    :param np.array y_test: optional 1D matrix of labels for the given testing
        data points.
    :returns Tuple[np.array, np.array]: The new processed (X_train, y_train)
        data if not test data was provided. Otherwise it returns the processed
        (X_train, y_train, X_test, y_test)
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
        return X_train, y_train, X_test, y_test
    return X_train, y_train


def replace_categorical_outputs(
    X_train,
    y_train,
    output_classes,
    X_test=None,
    y_test=None,
):
    """
    Simple scaling preprocessing function to replace categorical values in the
    given vector y_train with their numerical encodings.

    :param np.array X_train: 2D matrix of data points to be used for training.
    :param np.array y_train: 1D matrix of labels for the given training data
        points.
    :param List[OutputClass]: list of possible output categorical classes in
        vector y_train.
    :param np.array X_test: optional 2D matrix of data points to be used for
        testing.
    :param np.array y_test: optional 1D matrix of labels for the given testing
        data points.
    :returns Tuple[np.array, np.array]: The new processed (X_train, y_train)
        data if not test data was provided. Otherwise it returns the processed
        (X_train, y_train, X_test, y_test)
    """
    out_map = {
        c.name: c.encoding
        for c in output_classes
    }
    for i, val in enumerate(y_train):
        y_train[i] = out_map[val]
    if y_test is not None:
        for i, val in enumerate(y_test):
            y_test[i] = out_map[val]
    if X_test is not None:
        return (
            X_train,
            y_train.astype(np.int32),
            X_test,
            y_test.astype(np.int32),
        )

    return X_train, y_train.astype(np.int32)


################################################################################
## Exposed Methods
################################################################################


def get_data_configuration(dataset_name):
    """
    Gets the configuration of dataset with name `dataset_name` if it is a
    supported dataset (case insensitive). Otherwise a ValueError is thrown.

    :param str dataset_name:  The name of the dataset we want to fetch.
    :return DatasetDescriptor:  The configuration corresponding to the given
                              dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'artif-1':
        output_classes = (
            OutputClass(name='y0', encoding=0),
            OutputClass(name='y1', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='y',
        )
    if dataset_name == 'artif-2':
        output_classes = (
            OutputClass(name='y0', encoding=0),
            OutputClass(name='y1', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='y',
        )
    if dataset_name == 'mb-ge-er':
        output_classes = (
            OutputClass(name='negative', encoding=0),
            OutputClass(name='positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='ER_Expr',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )
    if dataset_name == 'breastcancer':
        output_classes = (
            OutputClass(name='M', encoding=0),
            OutputClass(name='B', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='diagnosis',
        )
    if dataset_name == 'iris':
        output_classes = (
            OutputClass(name='Setosa', encoding=0),
            OutputClass(name='Versicolor', encoding=1),
            OutputClass(name='Virginica', encoding=2),
        )

        # Helper method for preprocessing our data
        def preprocess_fun(X_train, y_train, X_test=None, y_test=None):
            return replace_categorical_outputs(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                output_classes=output_classes,
            )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=preprocess_fun,
            target_col='variety',
        )
    if dataset_name == 'letterrecognition':
        output_classes = (
            OutputClass(name='A', encoding=0),
            OutputClass(name='B-Z', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='letter',
        )

    if dataset_name == 'letterrecognitioncomplete':
        output_classes = [
            OutputClass(
                name=chr(i),
                encoding=i - ord('A'),
            ) for i in range(ord('A'), ord('Z') + 1, 1)
        ]

        def preprocess_fun(X_train, y_train, X_test=None, y_test=None):
            return replace_categorical_outputs(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                output_classes=output_classes,
            )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='letter',
            preprocessing=preprocess_fun,
        )

    if dataset_name == 'glassidentification':
        output_classes = [
            OutputClass(
                name='building_windows_float_processed',
                encoding=0,
            ),
            OutputClass(
                name='building_windows_non_float_processed',
                encoding=1,
            ),
            OutputClass(
                name='vehicle_windows_float_processed',
                encoding=2,
            ),
            OutputClass(
                name='containers',
                encoding=3,
            ),
            OutputClass(
                name='tableware',
                encoding=4,
            ),
            OutputClass(
                name='headlamps',
                encoding=5,
            ),
        ]
        def preprocess_fun(X_train, y_train, X_test=None, y_test=None):
            return replace_categorical_outputs(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                output_classes=output_classes,
            )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='Type',
            preprocessing=preprocess_fun,
        )

    if dataset_name == 'forestcovertype':
        output_classes = [
            OutputClass(
                name='Spruce/Fir',
                encoding=0,
            ),
            OutputClass(
                name='Lodgepole_Pine',
                encoding=1,
            ),
            OutputClass(
                name='Ponderosa_Pine',
                encoding=2,
            ),
            OutputClass(
                name='Cottonwood/Willow',
                encoding=3,
            ),
            OutputClass(
                name='Aspen',
                encoding=4,
            ),
            OutputClass(
                name='Douglas-fir',
                encoding=5,
            ),
            OutputClass(
                name='Krummholz',
                encoding=6,
            ),
        ]
        def preprocess_fun(X_train, y_train, X_test=None, y_test=None):
            return replace_categorical_outputs(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                output_classes=output_classes,
            )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='Cover_Type',
            preprocessing=preprocess_fun,
        )

    if dataset_name == 'mnist':
        output_classes = (
            OutputClass(name='0', encoding=0),
            OutputClass(name='1-9', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='digit',
        )
    if dataset_name == 'mnist-complete':
        output_classes = [
            OutputClass(name=str(i), encoding=i)
            for i in range(10)
        ]
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='digit',
        )
    if dataset_name == 'tcga-pancan':
        output_classes = (
            OutputClass(name='BRCA', encoding=0),
            OutputClass(name='KIRC', encoding=1),
            OutputClass(name='LAUD', encoding=2),
            OutputClass(name='PRAD', encoding=3),
            OutputClass(name='COAD', encoding=4),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='TCGA',
        )
    if dataset_name == 'mb-ge-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    if dataset_name == 'mb-clin-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    if dataset_name == 'mb-clin-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-clinp-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-clin-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-clinp-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )

    if dataset_name == 'mb_ge_cdh1_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )

    if dataset_name == 'mb-1004-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )

    if dataset_name == 'mb-imagevec5-6hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
            OutputClass(name='IDC+ILC', encoding=2),
            OutputClass(name='IDC-MUC', encoding=3),
            OutputClass(name='IDC-TUB', encoding=4),
            OutputClass(name='IDC-MED', encoding=5),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb-imagevec50-6hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
            OutputClass(name='IDC+ILC', encoding=2),
            OutputClass(name='IDC-MUC', encoding=3),
            OutputClass(name='IDC-TUB', encoding=4),
            OutputClass(name='IDC-MED', encoding=5),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb_imagevec50_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb-ge-6hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
            OutputClass(name='IDC+ILC', encoding=2),
            OutputClass(name='IDC-MUC', encoding=3),
            OutputClass(name='IDC-TUB', encoding=4),
            OutputClass(name='IDC-MED', encoding=5),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )

    if dataset_name == 'mb_imagevec50_er':
        output_classes = (
            OutputClass(name='ER Negative', encoding=0),
            OutputClass(name='ER Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb_imagevec50_dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    if dataset_name == 'miniboone':
        output_classes = (
            OutputClass(name='electron_neutrino', encoding=0),
            OutputClass(name='muon_neutrino', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            # preprocessing=unit_scale_preprocess,
            target_col='event',
        )

    if dataset_name == 'xor':
        output_classes = (
            OutputClass(name='0', encoding=0),
            OutputClass(name='1', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='xor',
        )

    if dataset_name == 'sarcos':
        return DatasetDescriptor(
            name=dataset_name,
            target_col='torque',
            regression=True,
        )

    if dataset_name == 'winequality':
        return DatasetDescriptor(
            name=dataset_name,
            target_col='quality',
            regression=True,
            preprocessing=unit_scale_preprocess,
        )

    if dataset_name == 'winequalityclassification':
        return DatasetDescriptor(
            name=dataset_name,
            target_col='quality',
            preprocessing=unit_scale_preprocess,
            output_classes=[
                OutputClass(name=str(i), encoding=i)
                for i in range(10)
            ]
        )

    if dataset_name == 'magic':
        output_classes = (
            OutputClass(name='hadron', encoding=0),
            OutputClass(name='gamma', encoding=1),
        )
        def preprocess_fun(X_train, y_train, X_test=None, y_test=None):
            result = replace_categorical_outputs(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                output_classes=output_classes,
            )
            return unit_scale_preprocess(*result)

        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=preprocess_fun,
            target_col='class',
        )

    if dataset_name == "partner-clinical":
        output_classes = (
            OutputClass(name='non-pCR', encoding=0),
            OutputClass(name='pCR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='pCR',
            feature_descriptors={
                None: RealDescriptor(),
                "Age": DiscreteNumericDescriptor(
                    list(range(1, 100)),
                    units="yrs",
                ),
                "Receptor_Status": TrivialCatDescriptor(
                    ["Negative", "Positive"]
                ),
                "Tumour_subtype": TrivialCatDescriptor([
                    "Dictal(NST)",
                    "Metaplastic",
                    "Medualiary",
                    "Apocrine",
                    "Mixed",
                ]),
                "Ductal_subtype": TrivialCatDescriptor(["No Ductal", "Ductal"]),
                "Grade": DiscreteNumericDescriptor([1, 2, 3]),
                "Largest_Clinical_Size": RealDescriptor(min_val=0, units="mm"),
                "T4_or_Inflammatory": TrivialCatDescriptor(["No", "Yes"]),
                "Clinical_Nodal_Involvement": TrivialCatDescriptor(
                    ["No", "Yes"]
                ),
                "Clinical_Stage": TrivialCatDescriptor([
                    "IA",
                    "IIA",
                    "IIIA",
                    "IIB",
                    "IIIB",
                    "IIIC",
                ]),
                "Smoking_category": TrivialCatDescriptor([
                    "Never",
                    "Current",
                    "Former",
                    "Unknown",
                ]),
                "BMI": RealDescriptor(min_val=0, units="kg/m^2"),
                "anthracycline": TrivialCatDescriptor(["No", "Yes"]),
                "PARTNER": TrivialCatDescriptor(["No", "Yes"]),
                "treatment_group": TrivialCatDescriptor(
                    ["Control", "Olaparib"]
                ),
                "TILs": RealDescriptor(min_val=0, max_val=1),
                "EGFR": TrivialCatDescriptor(["Negative", "Positive"]),
                "CK5_6": TrivialCatDescriptor(["Negative", "Positive"]),
                "ARIHC": RealDescriptor(min_val=0, max_val=100, units="%"),
                "pCR": RealDescriptor(["non-pCR", "pCR"]),
            }
        )

    if dataset_name == "partner-genomic":
        output_classes = (
            OutputClass(name='non-pCR', encoding=0),
            OutputClass(name='pCR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='pCR',
            preprocessing=unit_scale_preprocess,
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
                "pCR": RealDescriptor(["non-pCR", "pCR"]),
            }
        )

    # Else this is a big no-no
    raise ValueError(f'Invalid dataset name "{dataset_name}"')


def is_valid_dataset(dataset_name):
    """
    Determines whether the specified dataset name is valid dataset name within
    the supported datasets for experimentation.

    :param str dataset_name:  The name of the dataset we want to check
    :return bool: whether or not this is a valid dataset.
    """
    return dataset_name.lower() in list(map(
        lambda x: x.lower(),
        AVAILABLE_DATASETS,
    ))


