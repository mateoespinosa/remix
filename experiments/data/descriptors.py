import pandas as pd

################################################################################
## Feature Descriptor Classes
################################################################################


class FeatureDescriptor(object):
    """
    Base class to abstract annotations of a single feature such as its
    nature, its units, and its bounds.

    Useful for visualization purposes and pretty printing of data.
    """
    def __init__(self, units=None):
        self.units = units

    def is_normalized(self):
        return False

    def transform_to_numeric(self, x):
        # By default this is the identity value
        return x

    def transform_from_numeric(self, x):
        # By default this is the identity value
        return x

    def is_discrete(self):
        raise UnimplementedError("is_discrete(self)")

    def default_value(self):
        raise UnimplementedError("default_value(self)")

    def numeric_bounds(self):
        return (-float("inf"), float("inf"))

    def is_categorical(self):
        raise UnimplementedError("is_categorical(self)")


class RealDescriptor(FeatureDescriptor):
    """
    Class describing a feature whose range is the real numbers.
    """

    def __init__(
        self,
        max_val=float("inf"),
        min_val=-float("inf"),
        normalized=False,
        units=None,
    ):
        super(RealDescriptor, self).__init__(units=units)
        self.normalized = normalized
        self.max_val = max_val
        self.min_val = min_val

    def default_value(self):
        if self.max_val not in [float("inf"), -float("inf")] and (
            self.min_val not in [float("inf"), -float("inf")]
        ):
            return (self.max_val + self.min_val)/2
        elif self.max_val not in [float("inf"), -float("inf")]:
            return self.max_val
        elif self.min_val not in [float("inf"), -float("inf")]:
            return self.min_val
        return 0

    def numeric_bounds(self):
        return (self.min_val, self.max_val)

    def is_discrete(self):
        return False

    def is_categorical(self):
        return False

    def is_normalized(self):
        return self.normalized


class DiscreteNumericDescriptor(RealDescriptor):
    """
    Class describing a feature that is discrete in nature and restricted
    a list of possible values.
    """

    def __init__(self, values, units=None):
        super(DiscreteNumericDescriptor, self).__init__(units=units)
        self.values = sorted(values)
        if values:
            self.max_val = values[-1]
        else:
            self.max_val = float("inf")
        if values:
            self.min_val = values[0]
        else:
            self.min_val = -float("inf")

    def default_value(self):
        if self.values:
            return self.values[len(self.values)//2]
        return 0

    def is_discrete(self):
        return True


class DiscreteEncodingDescriptor(DiscreteNumericDescriptor):
    """
    Class describing a discrete feature together with some numerical encoding
    of each possible entry in this feature's set of values.
    """

    def __init__(self, encoding_map, units=None):
        self.encoding_map = encoding_map
        values = []
        self.inverse_map = {}
        self._default_value = None
        for datum_name, numeric_val in self.encoding_map.items():
            self._default_value = self._default_value or datum_name
            values.append(numeric_val)
            self.inverse_map[numeric_val] = datum_name

        super(DiscreteEncodingDescriptor, self).__init__(
            values=values,
            units=units,
        )

    def default_value(self):
        return self._default_value

    def is_categorical(self):
        return True

    def transform_to_numeric(self, x):
        if isinstance(x, (int)):
            # Then this is already the numeric encoding
            return x
        return self.encoding_map[x]

    def transform_from_numeric(self, x):
        if not isinstance(x, int):
            # Then this is already not a numeric encoding
            return x
        return self.inverse_map[x]


class TrivialCatDescriptor(DiscreteEncodingDescriptor):
    """
    This class describes a trivial categorical feature where every possible
    value it can take is encoded using its zero-indexed position in the list
    `vals`.
    """
    def __init__(self, vals, units=None):
        super(TrivialCatDescriptor, self).__init__(
            encoding_map=dict(
                zip(vals, range(len(vals)))
            ),
            units=units,
        )


################################################################################
## Helper Classes
################################################################################


class OutputClass(object):
    """
    Represents the conclusion of a given rule. Immutable and Hashable.
    Each output class has a name and its relevant integer encoding
    """

    def __init__(self, name: str, encoding: int):
        self.name = name
        self.encoding = encoding

    def __str__(self):
        return f'{self.name} (encoding {self.encoding})'

    def __eq__(self, other):
        return (
            isinstance(other, OutputClass) and
            (self.name == other.name) and
            (self.encoding == other.encoding)
        )

    def __hash__(self):
        return hash((self.name, self.encoding))


# Define a class that can be used to encapsulate all the information we will
# store from a given dataset.
class DatasetDescriptor(object):
    """
    Class abstracting a full annotated description of a given dataset.
    This annotation includes feature names, output classes (for classification
    tasks) and possible annotations on the nature and ranges of its input
    features.
    """
    def __init__(
        self,
        name="dataset",
        output_classes=None,
        n_features=None,
        target_col=None,
        feature_names=None,
        preprocessing=None,
        feature_descriptors=None,
        regression=False,
    ):
        self.name = name
        self.n_features = n_features
        self.output_classes = output_classes
        self.target_col = target_col
        self.feature_names = feature_names
        self.preprocessing = preprocessing
        self.feature_descriptors = feature_descriptors
        self.data = None
        self.X = None
        self.y = None
        self.regression = regression

    def process_dataframe(self, df):
        """
        Reads the dataset from the given dataframe and updates its missing
        descriptor values according to the provided dataset.

        :param pd.Dataframe df:  A pandas Dataframe describing the dataset
            corresponding to this dataset descriptor.
        """

        self.data = df
        # Set the target column, number of inputs, and feature names of our
        # dataset accordingly from the opened file if they were not provided
        self.target_col = self.target_col or (
            self.data.columns[-1]
        )
        self.n_features = self.n_features or (
            len(self.data.columns) - 1
        )
        self.feature_names = self.feature_names or (
            self.data.columns[:self.n_features]
        )
        if self.feature_descriptors is None:
            # Then we will assume they are arbitrary real numbers anyone
            # can set
            self.feature_descriptors = {
                None: RealDescriptor(
                    max_val=float("inf"),
                    min_val=-float("inf"),
                )
            }

        self.X = self.data.drop([self.target_col], axis=1).values
        self.y = self.data[self.target_col].values
        if (self.output_classes is None) and (not self.regression):
            out_classes = sorted(list(set(self.y)))
            self.output_classes = []
            for out_class in out_classes:
                self.output_classes.append(
                    OutputClass(name='{out_class}', encoding=out_class)
                )
        return self.X, self.y, self.data

    def read_data(self, data_path):
        # Read our dataset. This will be the first thing we will do:
        return self.process_dataframe(pd.read_csv(data_path, sep=','))

    def get_feature_ranges(self, feature_name):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].numeric_bounds()
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).numeric_bounds()

    def get_allowed_values(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None
        if feature_name in self.feature_descriptors:
            descriptor = self.feature_descriptors[feature_name]
            if descriptor.is_discrete():
                return descriptor.values
            return None
        return None

    def is_categorical(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None

        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].is_categorical()
        return False

    def is_discrete(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None

        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].is_discrete()
        return False

    def get_units(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None

        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].units
        return None

    def get_default_value(self, feature_name):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].default_value()
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).default_value()

    def transform_to_numeric(self, feature_name, x):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].transform_to_numeric(
                x
            )
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).transform_to_numeric(x)

    def transform_from_numeric(self, feature_name, x):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].transform_from_numeric(
                x
            )
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).transform_from_numeric(x)

    def is_normalized(self, feature_name):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].is_normalized()
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).is_normalized()
