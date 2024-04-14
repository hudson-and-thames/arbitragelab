"""
Abstract pair selector class.
"""

from abc import ABC
from abc import abstractmethod

import sys


class AbstractPairsSelector(ABC):
    """
    This is an abstract class for pairs selectors objects.
    It has abstract method select_pairs(), which needs to be implemented.
    """

    @abstractmethod
    def select_spreads(self):
        """
        Method which selects pairs based on some predefined criteria.
        """

        raise NotImplementedError('Must implement select_pairs() method.')

    @staticmethod
    def _print_progress(iteration, max_iterations, prefix='', suffix='', decimals=1, bar_length=50):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

        :param iteration: (int) Current iteration.
        :param max_iterations: (int) Maximum number of iterations.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """

        str_format = "{0:." + str(decimals) + "f}"
        # Calculate the percent completed.
        percents = str_format.format(100 * (iteration / float(max_iterations)))
        # Calculate the length of bar.
        filled_length = int(round(bar_length * iteration / float(max_iterations)))
        # Fill the bar.
        block = '█' * filled_length + '-' * (bar_length - filled_length)
        # Print new line.
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, block, percents, '%', suffix)),

        if iteration == max_iterations:
            sys.stdout.write('\n')
        sys.stdout.flush()
