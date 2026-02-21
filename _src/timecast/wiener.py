import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression


ZERO = 1e-4
ALT_STD = 0.1
BSTD_TYPE = "gaussian"


class wiener:
    def __init__(self):
        self.rng = default_rng()
        np.random.seed(0)
        self.drift = 0
        self.b_std = 0
        self.x_0 = 0
        self.d_thre = 1
        self.model_exist = False

    def projection_fit(self, feature_mat, label, verbose=False):
        """
        Estimate parameters for Wiener process with observation projection.

        Parameters
        ----------
        feature_mat : ndarray, shape (d, n)
        label : ndarray, shape (1, n)
        """
        self.model_exist = True
        label = label.copy()
        label[label == 0] = 1

        label_max = np.max(label)
        label_min = np.min(label)
        label_min = 1 + ZERO if label_min == 1 else label_min

        self.reg = LinearRegression().fit(feature_mat.T, label.T)
        self.proj = np.dot(label, np.linalg.pinv(feature_mat))
        diff_array = 1.0 / label

        mean_drift = np.mean(diff_array)
        if BSTD_TYPE == "gaussian":
            b_std = np.sqrt(np.mean((diff_array - mean_drift) ** 2))

        self.mean_drift = mean_drift
        self.min_drift = 1.0 / label_max
        self.max_drift = 1.0 / label_min
        self.b_std = b_std

        if b_std < ZERO:
            print(f"b_std is too small, set to {ALT_STD}")
            b_std = ALT_STD
            self.b_std = b_std

        if verbose:
            print(f"proj shape: {self.proj.shape}")
            print(f"mean_drift: {mean_drift}, b_std: {b_std}")

        return mean_drift, b_std

    def estimate_rul_val(self, feature, cur_state, value_refine=False, thre=1.0, reg=True):
        """
        Estimate RUL mean and variance.

        Parameters
        ----------
        feature : ndarray, shape (d, 1)
        cur_state : float
        """
        drift = self.get_drift(feature, reg=reg)

        if value_refine:
            drift = np.clip(drift, self.min_drift, thre - ZERO)

        mean = (thre - cur_state) / drift
        variance = (thre - cur_state) * self.b_std ** 2 / (drift ** 3)
        return mean, variance

    def likelihood_rul(self, feature, x, cur_state=0.0, thre=1.0, reg=True, refine=True):
        """Return likelihood for given RUL/lifetime value x."""
        drift = self.get_drift(feature, reg=reg)
        x = max(x, 1)

        if refine:
            drift = np.clip(drift, self.min_drift, thre - ZERO)

        if not self.model_exist:
            print("Model not fitted yet.")
            return 0.0

        return (
            (thre - cur_state)
            / np.sqrt(2 * np.pi * self.b_std ** 2 * x ** 3)
            * np.exp(
                -((thre - cur_state - drift * x) ** 2)
                / (2 * self.b_std ** 2 * x)
            )
        )

    def get_drift(self, feature, reg=True):
        if reg:
            return 1.0 / self.reg.predict(feature.T)[0, 0]
        return 1.0 / np.dot(self.proj, feature)[0, 0]

    def mae(self, feature, y, cur_state=0.0):
        mean, _ = self.estimate_rul_val(feature, cur_state)
        return abs(y - mean)

    def mse(self, feature, y, cur_state=0.0):
        mean, _ = self.estimate_rul_val(feature, cur_state)
        return (y - mean) ** 2

    def w_obj(self, feature, y, cur_state=0.0, thre=1.0):
        mean, _ = self.estimate_rul_val(feature, cur_state)
        y = max(y, 1)
        return (
            (1.0 / y - self.mean_drift) ** 2 / (self.b_std ** 2)
            + np.log(self.b_std ** 2)
            + (mean - y) ** 2
        )

    def update_mean_std(self, feature_mat, label, past_ns):
        """Online update of mean drift and b_std."""
        label = label.copy()
        label[label == 0] = 1
        diff_array = 1.0 / label
        add_mean_drift = np.mean(diff_array)
        add_ns = len(diff_array)
        add_var = np.mean((diff_array - add_mean_drift) ** 2)

        if BSTD_TYPE == "gaussian":
            all_mean = (self.mean_drift * past_ns + add_mean_drift * add_ns) / (past_ns + add_ns)
            delta_mean = self.mean_drift - add_mean_drift
            b_std = (
                self.b_std * past_ns
                + add_var * add_ns
                + (past_ns * add_ns / (past_ns + add_ns)) * delta_mean ** 2
            )
        self.mean_drift = all_mean
        self.b_std = b_std