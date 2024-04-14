"""
Tests functionality of Indexed Highlighter:
utils/indexed_highlight.py
"""

import warnings
import unittest
from unittest.mock import Mock, MagicMock

from arbitragelab.util.indexed_highlight import IndexedHighlight

class TestIndexedHighlight(unittest.TestCase):
    """
    Tests Indexed Highlighter class.
    """

    def setUp(self):
        """
        Tests Initial instantiation of the IndexedHighlighter class.
        """

        # This is here to hide 'deprecated in Matplotlib 3.1' warnings. The
        # functions mentioned here are needed for backward compatibility.
        warnings.simplefilter("ignore")

        placeholder_annotation = Mock()
        placeholder_annotation.xyann = [0, 0]

        artist = MagicMock(return_value=[])
        artist.color = "White"
        artist.visible = False
        artist.axes.annotate.return_value = placeholder_annotation
        artist.axes.figure.canvas.callbacks.callbacks = {'button_press_event': {}}

        indexed_highlighter = IndexedHighlight([artist])

        self.artist = artist
        self.indexed_highlighter = indexed_highlighter

    def test_assigned_highlights(self):
        """
        Tests size of the highlighter list.
        """

        self.assertEqual(len(self.indexed_highlighter.highlights), 1)

    def test_update_known_artist(self):
        """
        Tests the update function with a known artist inside the event object.
        """

        event = Mock()
        event.ind = [1]
        event.artist = self.artist
        event.artist.axes.xaxis.get_view_interval.return_value = [0, 0]
        event.artist.axes.yaxis.get_view_interval.return_value = [0, 0]
        event.mouseevent.xdata = 1
        event.mouseevent.ydata = 1

        bbox = Mock()
        bbox.corners.return_value = []

        annotation = Mock()
        annotation.get_window_extent.return_value = bbox

        self.indexed_highlighter.update(event, annotation)
        self.assertTrue(isinstance(self.indexed_highlighter.highlights[0], Mock))

    def test_update_unknown_artist(self):
        """
        Tests the update function with a unknown artist inside the event object.
        """

        bbox = Mock()
        bbox.corners.return_value = []

        annotation = Mock()
        annotation.get_window_extent.return_value = bbox

        new_event = Mock()
        new_event.ind = [1]
        new_event.artist.axes.xaxis.get_view_interval.return_value = [0, 0]
        new_event.artist.axes.yaxis.get_view_interval.return_value = [0, 0]
        new_event.mouseevent.xdata = 1
        new_event.mouseevent.ydata = 1

        self.indexed_highlighter.update(new_event, annotation)
        self.assertTrue(isinstance(self.indexed_highlighter.highlights[0], Mock))
