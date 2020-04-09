from tensorflow.keras import callbacks
import warnings

import utils


class MonitorLossCallback(callbacks.Callback):
    def __init__(self, monitor='loss', value=10000.0, verbose=1):
        super(callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        """
        Monitor loss and stop training for very large values of loss
        """
        current_loss = logs.get(self.monitor)
        if current_loss >= self.value:
            warnings.warn("Training loss is too large", RuntimeWarning)
            self.model.stop_training = True


class PredictCallback(callbacks.Callback):
    def __init__(self, test_data, test_labels, reverse_data_dictionary, n_epochs, next_compatible_tools, usg_scores):
        self.test_data = test_data
        self.test_labels = test_labels
        self.reverse_data_dictionary = reverse_data_dictionary
        self.precision = list()
        self.usage_weights = list()
        self.n_epochs = n_epochs
        self.next_compatible_tools = next_compatible_tools
        self.pred_usage_scores = usg_scores

    def on_epoch_end(self, epoch, logs={}):
        """
        Compute absolute and compatible precision for test data
        """
        if len(self.test_data) > 0:
            precision, usage_weights = utils.verify_model(self.model, self.test_data, self.test_labels, self.reverse_data_dictionary, self.pred_usage_scores)
            self.precision.append(precision)
            self.usage_weights.append(usage_weights)
            print("Epoch %d precision: %s" % (epoch + 1, precision))
            print("Epoch %d usage weights: %s" % (epoch + 1, usage_weights))
