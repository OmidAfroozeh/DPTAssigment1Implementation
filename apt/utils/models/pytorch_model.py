import random
from typing import Optional, Tuple

import numpy as np
from art.utils import check_and_transform_label_format

from sklearn.preprocessing import OneHotEncoder

from apt.utils.models import Model, ModelOutputType
from apt.utils.datasets import Dataset, OUTPUT_DATA_ARRAY_TYPE

from art.estimators.classification.pytorch import PyTorchClassifier as ArtPyTorchClassifier
import torch


class PyTorchModel(Model):
    """
    Wrapper class for pytorch models.
    """


class PyTorchClassifierWrapper(ArtPyTorchClassifier):
    """
        Wrapper class for pytorch classifier model.
        """

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
                Fit the classifier on the training set `(x, y)`.

                :param x: Training data.
                :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                          shape (nb_samples,).
                :param batch_size: Size of batches.
                :param nb_epochs: Number of epochs to use for training.
                :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
                """
        # Put the model in the training mode
        self._model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = torch.from_numpy(x_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self._model(i_batch)

                # Form the loss function
                loss = self._loss(model_outputs[-1], o_batch)  # lgtm [py/call-to-non-callable]

                loss.backward()

                self._optimizer.step()


class PyTorchClassifier(PyTorchModel):
    """
    Wrapper class for scikitlearn classification models.
    """

    def __init__(self, model: "torch.nn.Module", output_type: ModelOutputType, loss: "torch.nn.modules.loss._Loss",
                 input_shape: Tuple[int, ...], nb_classes: int, optimizer: Optional["torch.optim.Optimizer"] = None,
                 black_box_access: Optional[bool] = True, unlimited_queries: Optional[bool] = True, **kwargs):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param output_type: The type of output the model yields (vector/label only for classifiers,
                            value for regressors)
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param optimizer: The optimizer used to train the classifier.
        :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                                 Set to True if the model is only available via query (API) access, i.e.,
                                 only the outputs of the model are exposed, and False if the model internals
                                 are also available. Optional, Default is True.
        :param unlimited_queries: If black_box_access is True, this boolean indicates whether a user can perform
                                  unlimited queries to the model API or whether there is a limit to the number of
                                  queries that can be submitted. Optional, Default is True.
        """
        super().__init__(model, output_type, black_box_access, unlimited_queries, **kwargs)
        self._art_model = PyTorchClassifierWrapper(model, loss, input_shape, nb_classes, optimizer)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        """
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(train_data.get_labels().reshape(-1, 1))
        self._art_model.fit(train_data.get_samples(), y_encoded, **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model (class probabilities, if supported).
        """
        return self._art_model.predict(x.get_samples(), **kwargs)

    def score(self, test_data: Dataset, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        """
        pass
