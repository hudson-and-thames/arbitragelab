# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html
"""
This module houses the extension HighlightingDataCursor class to support cluster
by cluster highlighting.
"""

import matplotlib.pyplot as plt
from mpldatacursor import DataCursor, HighlightingDataCursor

from arbitragelab.util import segment

class IndexedHighlight(HighlightingDataCursor):
    """
    This class extends HighlightingDataCursor to add support for
    highlighting of cluster groups.
    """

    def __init__(self, axes, **kwargs):
        """
        Initializes the highlighting object for each AxesSubplot in a plot.
        """
        artists = axes

        kwargs['display'] = 'single'
        HighlightingDataCursor.__init__(self, artists, **kwargs)
        self.highlights = [self.create_highlight(artist) for artist in artists]
        plt.setp(self.highlights, visible=False)

        segment.track('IndexedHighlight')

    def update(self, event, annotation):
        """
        On each update event, this method will loop through all SubPlot objects
        and the group of points corresponding to the current selected object
        will be highlighted.
        """

        # Hide all other annotations
        plt.setp(self.highlights, visible=False)

        for i, artst in enumerate(self.artists):
            if event.artist is artst:
                self.highlights[i].set(visible=True)

        DataCursor.update(self, event, annotation)
