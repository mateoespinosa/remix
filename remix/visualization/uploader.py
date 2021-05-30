"""
File containing code for IO management in file upload/download in Flexx-based
GUIs.
"""

from flexx import flx, app


class FileUploader(flx.BaseButton):
    """
    Helper Flexx class to perform an (unsecured!) file upload.
    Inspired by discussion in issue: https://github.com/flexxui/flexx/issues/588
    """

    CSS = """
    .flx-FileUploader {
       padding: 0;
       box-sizing: border-box;
    }
    """

    DEFAULT_MIN_SIZE = 10, 28
    file_name = flx.StringProp('', settable=True)
    binary = flx.BoolProp(False, settable=True)

    def _render_dom(self):
        """
        Our uploader's DOM will have a simple button which will allow the
        user to use to input a file path and trigger the upload.
        """
        global document, FileReader

        # Main button will be instantiated here
        self.action_but = document.createElement(
            'input'
        )
        self.action_but.className = "flx-Button flx-BaseButton"
        # And also a input object which will do the actual uploading
        self.file = document.createElement('input')

        # Stile the button correctly
        self.action_but.type = 'button'
        self.action_but.style = (
            "min-width: 10px; max-width: 1e+09px; min-height: 28px;"
            "max-height: 1e+09px;flex-grow: 1; flex-shrink: 1;"
            "margin-left: 0px; margin: 0; box-sizing: border-box; width: 100%;"
        )
        self.action_but.value = self.text

        # Add a listener to its clocking
        self._addEventListener(
            self.action_but,
            'click',
            self.__start_loading,
            0,
        )

        # And make sure the file object is also set correctly with a listener
        # to any changes so that the file can be handled
        self.file.type = 'file'
        self.file.style = 'display: none'
        self.file.addEventListener(
            'change',
            self._handle_file,
        )

        # Finally, we will need a file reader object to do the actual
        # deserialization and upload.
        self.reader = FileReader()
        self.reader.onload = self.file_loaded
        self.reader.onloadstart = self.load_started
        self.reader.onloadend = self.load_ended
        self.reader.onerror = self.reading_error

        return [self.action_but, self.file]

    def __start_loading(self, *events):
        """
        Starts loading a given file by simulating a click in the input object.
        """
        self.file.click()

    @flx.reaction('disabled')
    def __disabled_changed(self, *events):
        """
        Change the disabled state of this widget.
        """
        if events[-1].new_value:
            self.action_but.setAttribute("disabled", "disabled")
        else:
            self.action_but.removeAttribute("disabled")

    def _handle_file(self):
        """
        Handle the given file by opening it and reading it as a blob or text
        depending on the requested type.
        """
        if self.file.files.length > 0:
            self.set_file_name(self.file.files[0].name)
            self.file_selected()
            if self.binary:
                self.reader.readAsArrayBuffer(self.file.files[0])
            else:
                self.reader.readAsText(self.file.files[0])

    @flx.emitter
    def file_loaded(self, event):
        """
        Once file has been fully loaded, emit the `file_loaded` event.
        """
        return {
            'filedata': event.target.result,
            'filename': self.file_name,
        }

    @flx.emitter
    def load_started(self, event):
        """
        Once file loading has begun, emit the `load_started` event.
        """
        return {}

    @flx.emitter
    def load_ended(self, event):
        """
        Once file loading has ended, emit the `load_ended` event.
        """
        return {}

    @flx.emitter
    def reading_error(self, event):
        """
        If an error was found, then emit the `reading_error` event.
        """
        return event

    @flx.emitter
    def file_selected(self):
        """
        If a file path has been selected, then emit the `file_selected` event.
        """
        return {
            'filename': self.file_name,
        }


if __name__ == '__main__':
    """
    Simple testing code for this uploader.
    """

    class App(flx.PyComponent):
        def init(self):
            with flx.VBox():
                self.uploader = FileUploader(text="Try me")
                self.text_box = flx.Label("", flex=1)

        @flx.reaction('uploader.file_loaded')
        def handle_file_upload(self, *events):
            self.text_box.set_html(events[-1]['filedata'])

    app.serve(App)
    app.start()
