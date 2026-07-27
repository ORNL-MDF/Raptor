import numpy as np
from dataclasses import dataclass


# The unlist_list transformer
def unlist_list(func):
    def wrapper_func(self, *args):
        args_np = [np.asarray(y, dtype=float) for y in args]
        res_np = func(self, *args_np)
        return [y.tolist() for y in res_np]

    return wrapper_func


@dataclass
class InputScaler:
    bounds: list[tuple[float, float]]

    @unlist_list
    def to_unit(self, x):
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return (x - lo) / (hi - lo + 1e-12)

    @unlist_list
    def from_unit(self, x):
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return x * (hi - lo) + lo


@dataclass
class ScalerLog1p:
    y_prescale: float = 1.0
    y_postscale: float = 1.0

    @unlist_list
    def scale(self, y, yerr):
        y_scale = np.log1p(y / self.y_prescale) / self.y_postscale
        yerr_scale = yerr / (y + self.y_prescale) / self.y_postscale
        return y_scale, yerr_scale

    @unlist_list
    def unscale(self, y_scale, yerr_scale):
        y = self.y_prescale * np.expm1(self.y_postscale * y_scale)
        yerr = yerr_scale * (y + self.y_prescale) * self.y_postscale
        return y, yerr


@dataclass
class ScalerOutputFocus:
    y_low: float = 0.5
    y_high: float = 1.5
    focus: float = 1.0

    def params(self):
        y_mean = (self.y_low + self.y_high) / 2.0
        y_diff = (1 / self.focus) * (self.y_high - self.y_low) / 2.0
        return y_mean, y_diff

    @unlist_list
    def scale(self, y, yerr):
        y_mean, y_diff = self.params()
        y_norm = (y - y_mean) / y_diff
        yerr_norm = yerr / y_diff
        y_scale = np.asinh(y_norm)
        yerr_scale = 1 / np.sqrt(1 + y_norm**2) * yerr_norm
        return y_scale, yerr_scale

    @unlist_list
    def unscale(self, y_scale, yerr_scale):
        y_mean, y_diff = self.params()
        y_norm = np.sinh(y_scale)
        yerr_norm = np.sqrt(1 + y_norm**2) * yerr_scale
        y = y_diff * y_norm + y_mean
        yerr = y_diff * yerr_norm
        return y, yerr


@dataclass
class ScalerOutputFocusLog:
    y_low: float = 0.5
    y_high: float = 1.5
    focus: float = 1.0

    def params(self):
        ly_low = np.log(self.y_low)
        ly_high = np.log(self.y_high)
        y_mean = (ly_low + ly_high) / 2.0
        y_diff = (1 / self.focus) * (ly_high - ly_low) / 2.0
        return y_mean, y_diff

    @unlist_list
    def scale(self, y, yerr):
        logy = np.log(y)
        logyerr = yerr / y
        y_mean, y_diff = self.params()
        y_norm = (logy - y_mean) / y_diff
        yerr_norm = logyerr / y_diff
        y_scale = np.asinh(y_norm)
        yerr_scale = 1 / np.sqrt(1 + y_norm**2) * yerr_norm
        return y_scale, yerr_scale

    @unlist_list
    def unscale(self, y_scale, yerr_scale):
        y_mean, y_diff = self.params()
        y_norm = np.sinh(y_scale)
        yerr_norm = np.sqrt(1 + y_norm**2) * yerr_scale
        logy = y_diff * y_norm + y_mean
        logyerr = y_diff * yerr_norm
        y = np.exp(logy)
        yerr = logyerr * y
        return y, yerr


SCALER_REGISTRY = {
    "log1p": ScalerLog1p,
    "output_focus": ScalerOutputFocus,
    "output_focus_log": ScalerOutputFocusLog,
}
