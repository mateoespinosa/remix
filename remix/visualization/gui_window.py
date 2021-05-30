"""
This file contains our main class for a given window of our visualizer.
"""

from flexx import flx


class RemixWindow(flx.PyComponent):
    """
    Default class to be implemented for each window in our interactive
    visualization tool.
    """

    def init(self):
        super().init()

    @flx.emitter
    def ruleset_update(self, event):
        """
        Indicates that the shared application-wide rule set has been updated and
        produces an event describing the update that was made to the rule set.
        :param Dict[str, Any] event: Dictionary describing the update event.

        :return Dict[str, Any] event: Dictionary describing the update event.
        """
        return event

    @flx.action
    def reset(self):
        """
        Resets the entire window so that it updates itself based on the new
        state of the shared application-wide rule set.
        """
        pass

    @flx.action
    def perform_update(self, event):
        """
        Updates this window to take into account the changes made to the shared
        application-wide rule set as indicated by the given Flexx event
        dictionary.

        :param Dict[str, Any] event: Dictionary describing the update event.
        """
        # By default, this will trigger a whole reset
        self.reset()
