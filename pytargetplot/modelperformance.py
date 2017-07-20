#!/usr/bin/env python3.6
# coding: utf-8

"""Model performance."""


import json
from collections import OrderedDict
import pandas
import matplotlib.pyplot as plt

from pytargetplot import uncertainties, TargetPlot


class ModelPerformanceError(Exception):
    """Model performance exception."""
    pass


class ModelPerformance(object):
    """Model performance."""

    def __init__(self, df, param,
                 field_obs='obs', field_mod='mod', field_u='u'):
        """Model performance.

        :param df: pandas.DataFrame with data.
        :param param: parameter.
        :param field_obs: name of observation data field.
        :param field_mod: name of modelled data field.
        :param field_u: name of observation uncertainty data field.
        """
        self.df = df.copy()
        self.param = param
        self.field_obs = field_obs
        self.field_mod = field_mod
        self.field_u = field_u
        if self.field_obs not in self.df.columns:
            raise ModelPerformanceError("df must contains observation data !")
        if self.field_mod not in self.df.columns:
            raise ModelPerformanceError("df must contains modelled data !")
        if self.field_u not in self.df.columns:
            self._compute_obs_uncertainty()

    def _compute_obs_uncertainty(self):
        """Compute observation uncertainty."""
        u_cfg = uncertainties[self.param]
        self.df[self.field_u] = (u_cfg.k
                                 * u_cfg.uRV_r
                                 * ((1 - u_cfg.alpha)
                                    * self.df[self.field_obs] ** 2
                                    + u_cfg.alpha * u_cfg.RV ** 2)
                                 ** 0.5)

    @property
    def n(self):
        """Number of obs/mod data."""
        return len(self.df)

    @property
    def u(self):
        """U: observation uncertainty."""
        u = (self.df[self.field_u] ** 2).mean() ** 0.5
        return u

    @property
    def mean_obs(self):
        """Average observed value."""
        return self.df[self.field_obs].mean()

    @property
    def mean_mod(self):
        """Average modelled value."""
        return self.df[self.field_mod].mean()

    @property
    def rmse(self):
        """RMSE: Root Mean Square Error."""
        rmse = ((self.df[self.field_obs]
                 - self.df[self.field_mod]) ** 2).mean() ** 0.5
        return rmse

    @property
    def r(self):
        """R: correlation coefficient."""
        p1 = ((self.df[self.field_mod] - self.mean_mod) *
              (self.df[self.field_obs] - self.mean_obs)).sum()
        p2a = ((self.df[self.field_mod] - self.mean_mod) ** 2).sum() ** 0.5
        p2b = ((self.df[self.field_obs] - self.mean_obs) ** 2).sum() ** 0.5
        r = p1 / (p2a * p2b)
        return r

    @property
    def bias(self):
        """Bias."""
        bias = self.mean_mod - self.mean_obs
        return bias

    @property
    def nmb(self):
        """NMB: Normalised Mean Bias."""
        nmb = self.bias / self.mean_obs
        return nmb

    @property
    def sd_obs(self):
        """Standard deviation of observed values."""
        sd_obs = ((self.df[self.field_obs] - self.mean_obs) ** 2).mean() ** 0.5
        return sd_obs

    @property
    def sd_mod(self):
        """Standard deviation of modelled values."""
        sd_mod = ((self.df[self.field_mod] - self.mean_mod) ** 2).mean() ** 0.5
        return sd_mod

    @property
    def nmsd(self):
        """NMSD: Normalised Mean Standard Deviation."""
        nmsd = (self.sd_mod - self.sd_obs) / self.sd_obs
        return nmsd

    @property
    def rmsu(self):
        """RMSu: quadratic mean of the measurement uncertainty."""
        rmsu = (self.df[self.field_u] ** 2).mean() ** 0.5
        return rmsu

    @property
    def mqo(self):
        """MQO: Model Quality Objective.

        MQO <= 0.5:
          The model results are within the range of observation uncertainty.
          Model are closer to the true value.

        0.5 < MQO <= 1:
          Model results could still be closer to the true value than
          the observation.

        MQO > 1:
          Observation and model uncertainty ranges do not overlap.
        """
        mqo = 0.5 * self.rmse / self.rmsu
        return mqo

    @property
    def crmse(self):
        """CRMSE."""
        crmse = (((self.df[self.field_mod] - self.mean_mod) -
                 (self.df[self.field_obs] - self.mean_obs)) ** 2).mean()
        crmse = crmse ** 0.5
        return crmse

    @property
    def target_plot_coords(self):
        """Target Plot coordinates (x, y)."""
        # X
        x = self.crmse / (2 * self.u)
        ratio_crmse = (abs(self.sd_mod - self.sd_obs) /
                       (self.sd_obs * (2 * (1 - self.r)) ** 0.5))
        if ratio_crmse < 1:
            x *= -1

        # Y
        y = self.bias / (2 * self.u)

        return x, y

    def create_target_plot(self, fn=None, title=None):
        """Create and save a Target Plot.

        :param fn: path of figure to create.
        :param title: title of figure.
        """
        tp = TargetPlot()
        tp.plot(*self.target_plot_coords, 'o')
        if title:
            tp.title(title)
        if fn:
            tp.savefig(fn)
        else:
            plt.show()

    def num_of_hits(self, limit):
        """Number of hits (event forecast and observed occur).

        :param limit: limit.
        :return: int.
        """
        return len(
            self.df.query("{self.field_obs} > {limit} & "
                          "{self.field_mod} > {limit}".format(**locals())))

    def num_of_misses(self, limit):
        """Number of misses (event occur but not forecast).

        :param limit: limit.
        :return: int.
        """
        return len(
            self.df.query("{self.field_obs} > {limit} & "
                          "{self.field_mod} <= {limit}".format(**locals())))

    def num_of_false_alarms(self, limit):
        """Number of false alarms (event did not occur but it was forecast).

        :param limit: limit.
        :return: int.
        """
        return len(
            self.df.query("{self.field_obs} <= {limit} & "
                          "{self.field_mod} > {limit}".format(**locals())))

    def num_of_correct_negatives(self, limit):
        """Number of correct negatives (event did not occur and not forecast).

        :param limit: limit.
        :return: int.
        """
        return len(
            self.df.query("{self.field_obs} <= {limit} & "
                          "{self.field_mod} <= {limit}".format(**locals())))

    def accuracy(self, limit):
        """Model accuracy of event.

        :param limit: limit.
        :return: float between 0 and 1.
        """
        return (self.num_of_hits(limit) +
                self.num_of_correct_negatives(limit)) / self.n

    def probability_of_detection(self, limit):
        """Probability of detection of an event.

        :param limit: limit.
        :return: float between 0 and 1.
        """
        if self.num_of_hits(limit) + self.num_of_misses(limit):
            return (self.num_of_hits(limit) /
                    (self.num_of_hits(limit) + self.num_of_misses(limit)))
        else:
            return None

    def false_alarm_ratio(self, limit):
        """False alarm ratio of an event.

        :param limit: limit.
        :return: float between 0 and 1.
        """
        if self.num_of_hits(limit) + self.num_of_false_alarms(limit):
            return (self.num_of_false_alarms(limit) /
                    (self.num_of_hits(limit) + self.num_of_false_alarms(limit)))
        else:
            return None

    def contingency_table(self, limit):
        """Contingency table.

        :param limit: limit.
        :return: str.
        """
        mx = max(len(str(self.num_of_hits(limit))),
                 len(str(self.num_of_misses(limit))),
                 len(str(self.num_of_false_alarms(limit))),
                 len(str(self.num_of_correct_negatives(limit))))
        return ("      {a:{mx}} | {b:{mx}}\n"
                " obs  {sep}-+-{sep}\n"
                "      {c:{mx}} | {d:{mx}}\n"
                "      {esp}mod").format(
            a=self.num_of_false_alarms(limit),
            b=self.num_of_hits(limit),
            c=self.num_of_correct_negatives(limit),
            d=self.num_of_misses(limit),
            mx=mx, sep=mx * '-', esp=mx * ' ')

    @property
    def summary(self):
        """Resume of statistics.

        :return: dict.
        """
        x, y = self.target_plot_coords

        indics = OrderedDict()
        indics['Nb'] = self.n
        indics['Mean obs'] = self.mean_obs
        indics['Mean mod'] = self.mean_mod
        indics['U'] = self.u
        indics['RMSE'] = self.rmse
        indics['R'] = self.r
        indics['Bias'] = self.bias
        indics['NMB'] = self.nmb
        indics['SD obs'] = self.sd_obs
        indics['SD mod'] = self.sd_mod
        indics['NMSD'] = self.nmsd
        indics['RMSu'] = self.rmsu
        indics['MQO'] = self.mqo
        indics['CRMSE'] = self.crmse
        indics['Target Plot X'] = x
        indics['Target Plot Y'] = y

        return indics

    def to_pandas(self, title=None):
        """Export in pandas DataFrame.

        :param title: title of dataframe.
        :return: pandas.DataFrame.
        """
        indics = self.summary
        df = pandas.DataFrame(
            OrderedDict(((k, [v]) for k, v in indics.items()))).T
        if title:
            df.columns = [title, ]
        else:
            df.columns = ['', ]
        return df

    def __str__(self):
        """Table representation of statistics.

        :return: string.
        """
        df = self.to_pandas()
        return df.to_string(float_format=lambda x: format(x, '.3f'))

    def to_json(self):
        """Export in JSON format.

        :return: string.
        """
        return json.dumps(self.summary, indent=True)

    def to_dict(self):
        """Export as dictionnary.

        :return: dict.
        """
        return self.summary
