#!/usr/bin/env python3.6
# coding: utf-8

"""Air quality observation uncertainty."""


class ObsUncertainty(object):
    """Air quality observation uncertainty."""

    def __init__(self, **kw):
        """Air quality bservation uncertainty parameters."""
        for k, v in kw.items():
            self.__dict__[k] = v


uncertainties = {
    # source: Delta Tool Guide v5.2
    'NO2': ObsUncertainty(agreg='h', k=2, uRV_r=0.120, RV=200, alpha=0.04),
    'PM10': ObsUncertainty(agreg='d', k=2, uRV_r=0.140, RV=50, alpha=0.018),
    'PM2.5': ObsUncertainty(agreg='d', k=2, uRV_r=0.180, RV=25, alpha=0.05),
    'PM25': ObsUncertainty(agreg='d', k=2, uRV_r=0.180, RV=25, alpha=0.05),
    'O3': ObsUncertainty(agreg='h', k=1.4, uRV_r=0.090, RV=120, alpha=0.620),
}
