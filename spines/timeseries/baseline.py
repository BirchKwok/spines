from spines.timeseries.ts_toolsets import _split_sequences, _split_arrays
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from spines.timeseries.densenet import net
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class UnivariateRegression:
    def __init__(self, data, x_col, y_col, window_size, pred_days):
        assert isinstance(data, pd.DataFrame) is True
        self.y_pred_ = None
        self.data = data
        self.x_col, self.y_col, self.window_size, self.pred_days = x_col, y_col, window_size, pred_days
        self.model = None
        self.x_train_, self.y_train_, self.x_test_, self.y_test_ = None, None, None, None

    def _train_test_split(self, train_size=0.9, random_state=None):
        if train_size > 1 - round((self.window_size+self.pred_days+10) / len(self.data), 2):
            train_size = 1 - round((self.window_size+self.pred_days+10) / len(self.data), 2)
        return train_test_split(
            self.data[self.x_col], self.data[self.y_col], train_size=train_size,
            random_state=random_state, shuffle=False
        )

    def _generate_data(self):
        x_train, x_test, y_train, y_test = self._train_test_split(random_state=666)
        if self.pred_days == 1:
            self.x_train_, self.y_train_ = _split_arrays(x_train, y_train,
                                                         window_size=self.window_size, pred_days=self.pred_days)
            self.x_test_, self.y_test_ = _split_arrays(x_test, y_test,
                                                       window_size=self.window_size, pred_days=self.pred_days)
        else:
            self.x_train_, self.y_train_ = _split_sequences(x_train, y_train,
                                                            window_size=self.window_size, pred_days=self.pred_days)
            self.x_test_, self.y_test_ = _split_sequences(x_test, y_test,
                                                          window_size=self.window_size, pred_days=self.pred_days)

    def fit(self, callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=100, patience=100,
                                                            restore_best_weights=True), verbose='auto'):
        self._generate_data()
        if self.pred_days == 1:
            xgb = XGBRegressor()
            xgb.fit(self.x_train_, self.y_train_)
            self.model = xgb
        else:
            tf.random.set_seed(1024)
            np.random.seed(1024)

            input_shape = self.window_size

            callback = callback

            tf.keras.backend.clear_session()

            self.model = net([input_shape], output_nums=self.pred_days)

            self.model.summary()

            history = self.model.fit(x=self.x_train_, y=self.y_train_,
                                     validation_data=(self.x_test_, self.y_test_),
                                     epochs=1000,
                                     verbose=verbose,
                                     batch_size=20 if len(self.x_train_) < 800 else len(self.x_train_) // 40,
                                     callbacks=[callback]
                                     )

    def predict(self):
        assert self.model is not None
        if self.pred_days == 1:
            self.y_pred_ = self.model.predict(self.x_test_)
        else:
            self.y_pred_ = []
            for i in range(len(self.x_test_)):
                self.y_pred_.append(np.squeeze(self.model.predict(self.x_test_[i].reshape(1, -1, 1))))

        return self.y_pred_

    def plot_predict(self, nums_show=5):
        assert self.y_pred_ is not None, "Must to predict first, and then plot the figure."

        if self.pred_days == 1:
            plt.figure(figsize=(12, 8))
            textstr = '\n'.join([
                rf'r2 : {round(r2_score(self.y_test_, self.y_pred_), 2)}',
                rf'mae: {round(mean_absolute_error(self.y_test_, self.y_pred_), 2)}',
                rf'mape: {round(mean_absolute_percentage_error(self.y_test_, self.y_pred_), 2)}'
            ])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, max(self.y_test_), textstr, fontsize=14,
                     verticalalignment='top', bbox=props)
            plt.plot(range(len(self.y_pred_)), self.y_test_, label='true values')
            plt.plot(range(len(self.y_pred_)), self.y_pred_, label='predict values')
            plt.legend()
            plt.show()
        else:
            max_nums = len(self.y_test_)

            if nums_show > max_nums:
                nums_show = max_nums

            for i in range(nums_show):
                print(f"{i} picture.")
                plt.figure(figsize=(12, 8))
                textstr = '\n'.join([
                    rf'r2 : {round(r2_score(self.y_test_[i], np.squeeze(self.y_pred_[i])), 2)}',
                    rf'mae: {round(mean_absolute_error(self.y_test_[i], np.squeeze(self.y_pred_[i])), 2)}',
                    rf'mape: {round(mean_absolute_percentage_error(self.y_test_[i], np.squeeze(self.y_pred_[i])), 2)}'
                ])
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.text(0.05, max(self.y_test_[i]), textstr, fontsize=14,
                         verticalalignment='top', bbox=props)
                plt.plot(range(len(np.squeeze(self.y_pred_[i]))), self.y_test_[i], label='true values')
                plt.plot(range(len(np.squeeze(self.y_pred_[i]))), np.squeeze(self.y_pred_[i]), label='predict values')
                plt.legend()
                plt.show()

    def save_model(self, path):
        assert self.model is not None and self.pred_days is not None
        if self.pred_days == 1:
            self.model.save_model(path+'_xgboost_model')
        else:
            self.model.save(path+f'_keras_model_window_size_{self.window_size}')

        print('Model saved.')

    def load_model(self, path):
        raise NotImplementedError("Not implemented.")


class MultivariateRegression:
    def __init__(self):
        raise NotImplementedError("Not implemented.")

    def train(self):
        raise NotImplementedError("Not implemented.")

    def predict(self):
        raise NotImplementedError("Not implemented.")

    def plot_predict(self):
        raise NotImplementedError("Not implemented.")

    def save_model(self):
        raise NotImplementedError("Not implemented.")

    def load_model(self):
        raise NotImplementedError("Not implemented.")