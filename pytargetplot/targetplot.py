#!/usr/bin/env python3.6
# coding: utf-8

"""Target plot."""


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class TargetPlot(object):
    """Target Plot."""

    def __init__(self):
        """Target Plot."""
        # Init plot
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, aspect='equal')

        plt.plot([-2, 2], [0, 0], '-k', alpha=0.2)
        plt.plot([0, 0], [-2, 2], '-k', alpha=0.2)
        plt.plot([-2, 2], [-2, 2], '-k', alpha=0.2)
        plt.plot([-2, 2], [2, -2], '-k', alpha=0.2)

        plt.xlim(-2, 2)
        plt.xticks([-2, -1, 0, 1, 2])
        plt.xlabel("CRMSE")
        plt.ylim(-2, 2)
        plt.yticks([-2, -1, 0, 1, 2])
        plt.ylabel("BIAS")

        plt.annotate('T=1', xy=(0.05, 1.05), alpha=0.4)
        plt.annotate('T=.5', xy=(0.05, 0.55), alpha=0.4)

        plt.annotate('R', xy=(-1.9, -0.2), alpha=0.6)
        plt.annotate('SD', xy=(1.7, -0.2), alpha=0.6)
        plt.annotate('BIAS > 0', xy=(0, 1.8), alpha=0.6, ha='center')
        plt.annotate('BIAS < 0', xy=(0, -1.9), alpha=0.6, ha='center')

        circles = [
            mpatches.Circle((0, 0), 1, facecolor='green', edgecolor='none',
                            alpha=0.1),
            mpatches.Circle((0, 0), 1, fill=False, alpha=0.25, linestyle='-'),
            mpatches.Circle((0, 0), 0.5, fill=False, alpha=0.25,
                            linestyle='--'),
        ]
        for circle in circles:
            ax.add_patch(circle)

    @staticmethod
    def plot(x, y, style='o', **kwargs):
        """Add points.

        :param x: int, float, list of values or array.
        :param y: int, float, list of values or array.
        :param style: style of point.
        :param kwargs: kwargs of matplotlib.pyplot.plot function.
        """
        plt.plot(x, y, style, **kwargs)

    @staticmethod
    def title(title):
        """Add title.

        :param title: title.
        """
        plt.title(title)

    @staticmethod
    def savefig(fn):
        """Save figure into file.

        :param fn: path of file.
        """
        plt.savefig(fn)

    @staticmethod
    def __del__():
        plt.clf()
        plt.close()
