"""
Helper utilities common to several rule extractors.
"""

import pandas as pd
import scipy.special as activation_fns
import tensorflow.keras.models as keras


################################################################################
## Helper Classes
################################################################################

class ModelCache(object):
    """
    Represents trained neural network model. Used as a cache mechanism for
    storing intermediate activation values of an executed model.

    It index layers by indices and activations within a layer by names where:
        - The name of the i-th activation of the j-th hidden layer is given
           by "h_j_i"
        - The name of the i-th input feature activation is given by
          feature_names[i] if feature_names was given or h_0_i otherwise.
        - The name of the i-th output activation is given by
          output_class_names[i] if output_class_names was given or h_{d+1}_i
          otherwise (where d is the number of hidden layers in the network).
    """

    def __init__(
        self,
        keras_model,
        train_data,
        last_activation=None,
        feature_names=None,
        output_class_names=None,
    ):
        self._model = keras_model

        # Keeps in memory a map between layer ID and the activations it
        # generated when we processed the given training data
        self._activation_map = {}
        self._feature_names = feature_names
        self._output_class_names = output_class_names

        self._compute_layerwise_activations(
            train_data=train_data,
            last_activation=last_activation,
        )

    def __len__(self):
        """
        Returns the number of layers in this cache.
        """
        return len(self._model.layers)

    def _compute_layerwise_activations(self, train_data, last_activation=None):
        """
        Store sampled activations for each layer in dataframe which can then
        be used for quick access.
        """

        # Run the network once with the whole data, and pick up intermediate
        # activations

        feature_extractor = keras.Model(
            inputs=self._model.inputs,
            outputs=[layer.output for layer in self._model.layers]
        )
        # Run this model which will output all intermediate activations
        all_features = feature_extractor.predict(train_data)

        # And now label each intermediate activation using our
        # h_{layer}_{activation} notation
        for layer_index, (layer, activation) in enumerate(zip(
            self._model.layers,
            all_features,
        )):
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                if len(out_shape) == 1:
                    # Then we will allow degenerate singleton inputs
                    [out_shape] = out_shape
                else:
                    # Else this is not a sequential model!!
                    raise ValueError(
                        f"We encountered some branding in input model with "
                        f"layer at index {layer_index}"
                    )

            # If it is the first layer and we were given a list of feature
            # names, then let's make sure this list at least makes sense
            if (layer_index == 0) and (self._feature_names is not None):
                if len(self._feature_names) != activation.shape[-1]:
                    raise ValueError(
                        f"Expected input DNN to have {len(self._feature_names)}"
                        f" activations in it as we were given that many "
                        f"feature names but instead it has "
                        f"{activation.shape[-1]} input features."
                    )

            # Similarly, if it is the last layer and we were given a list of
            # output class names, let's check that these are correct
            if (layer_index == (len(self) - 1)) and (
                self._output_class_names is not None
            ):
                if len(self._output_class_names) != activation.shape[-1]:
                    raise ValueError(
                        f"Expected input DNN to have "
                        f"{len(self._output_class_names)} output activations "
                        f"in it as we were given that many "
                        f"output class names but instead it has "
                        f"{activation.shape[-1]} output activations."
                    )

            # Now time to name the different activations in this layer
            activation_labels = []
            for i in range(out_shape[-1]):
                if (layer_index == 0) and (self._feature_names is not None):
                    activation_labels.append(self._feature_names[i])
                elif (layer_index == (len(self) - 1)) and (
                    self._output_class_names is not None
                ):
                    activation_labels.append(self._output_class_names[i])
                else:
                    # Otherwise we will always use 'h_{layer_idx}_{act_idx}'
                    # for all unnamed activations
                    activation_labels.append(f'h_{layer_index}_{i}')

            # For the last layer, let's make sure it is turned into a
            # probability distribution in case the operation was merged into
            # the loss function. This is needed when the last activation (
            # e.g., softmax) is merged into the loss function (
            # e.g., softmax_cross_entropy).
            if last_activation and (layer_index == (len(self) - 1)):
                if last_activation == "softmax":
                    activation = activation_fns.softmax(activation, axis=-1)
                elif last_activation == "sigmoid":
                    # Else time to use sigmoid function here instead
                    activation = activation_fns.expit(activation)
                else:
                    raise ValueError(
                        f"We do not support last activation {last_activation}"
                    )

            self._activation_map[layer_index] = pd.DataFrame(
                data=activation,
                columns=activation_labels,
            )

    def get_layer_activations(self, layer_index):
        """
        Return activation values given layer index
        """
        result = self._activation_map[layer_index]
        return result

    def get_num_activations(self, layer_index):
        """
        Return the number of activations for the layer at the given index.
        """
        return self._activation_map[layer_index].shape[-1]
