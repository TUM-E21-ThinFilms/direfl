# Copyright (C) 2006-2010, University of Maryland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: James Krycka

import wx
import os
import numpy as np

import matplotlib

# Disable interactive mode so that plots are only updated on show() or draw().
# Note that the interactive function must be called before selecting a backend
# or importing pyplot, otherwise it will have no effect.
matplotlib.interactive(False)

# Specify the backend to use for plotting and import backend dependent classes.
# Note that this must be done before importing pyplot to have an effect.
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

# The Figure object is used to create backend-independent plot representations.
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Wx-Pylab magic ...
from matplotlib import _pylab_helpers
from matplotlib.backend_bases import FigureManagerBase

from wx.lib.wordwrap import wordwrap

from images import getOpenBitmap
from input_list import ItemListInput

# Specify desired initial window size (if physical screen size permits)
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 900
TASKBAR_HEIGHT = 32

REFL_FILES = "Refl files (*.refl)|*.refl"
DATA_FILES = "Data files (*.dat)|*.dat"
TEXT_FILES = "Text files (*.txt)|*.txt"
ALL_FILES = "All files (*.*)|*.*"

#==============================================================================

class AppFrame(wx.Frame):
    """This class implements the top-level frame for the applicaton."""

    def __init__(self, parent=None, id=wx.ID_ANY, title="",
                 pos=wx.DefaultPosition, size=(800, 600), name="AppFrame"
                ):
        wx.Frame.__init__(self, parent, id, title, pos, size, name=name)

        # Create a panel for the frame.  This will be the only child panel of
        # the frame and it inherits its size from the frame which is useful
        # during resize operations (as it provides a minimal size to sizers).
        self.panel = wx.Panel(self, wx.ID_ANY, style=wx.RAISED_BORDER)
        #self.panel = wx.Panel(self, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.panel.SetBackgroundColour("WHITE")

        # Display a splash screen.
        self.display_splash_screen()

        # Initialize the menu bar.
        self.add_menubar()

        # Initialize the tool bar.
        self.add_toolbar()

        # Initialize the status bar.
        self.add_statusbar([-3, -2, -1, -1, -1])

        # Initialize the notebook bar.
        self.add_notebookbar()

        # Uncomment call to Fit() to reduce the frame to minimum required size.
        # Comment out the call to keep the frame at its initial size.
        #self.Fit()


    def display_splash_screen(self):
        """Display the splash screen.  It will exactly cover the main frame."""

        x, y = self.GetSizeTuple()
        curdir = os.path.dirname(os.path.realpath(__file__))

        image = wx.Image(os.path.join(curdir, "splash.png"),
                         wx.BITMAP_TYPE_PNG)
        image.Rescale(x, y, wx.IMAGE_QUALITY_HIGH)
        bm = image.ConvertToBitmap()
        # bug? - wx.SPLASH_NO_CENTRE seems to ignore pos parameter; uses (0, 0)
        wx.SplashScreen(bitmap=bm,
                        #splashStyle=wx.SPLASH_NO_CENTRE|wx.SPLASH_TIMEOUT,
                        splashStyle=wx.SPLASH_CENTRE_ON_SCREEN|wx.SPLASH_TIMEOUT,
                        milliseconds=5000,
                        pos=self.GetPosition(),
                        parent=None, id=wx.ID_ANY)
        wx.Yield()


    def add_menubar(self):
        """Create a menu bar, menus, and menu options. """

        # Create the menubar.
        mb = wx.MenuBar()

        # Add a 'File' menu to the menu bar and define its options.
        menu1 = wx.Menu()

        load_id = menu1.Append(wx.ID_ANY, "&Load Data Files ...")
        self.Bind(wx.EVT_MENU, self.OnLoadData, load_id)

        menu1.AppendSeparator()

        load_id = menu1.Append(wx.ID_ANY, "&Load Model ...")
        self.Bind(wx.EVT_MENU, self.OnLoadModel, load_id)
        save_id = menu1.Append(wx.ID_ANY, "&Save Model ...")
        self.Bind(wx.EVT_MENU, self.OnSaveModel, save_id)

        menu1.AppendSeparator()

        exit_id = menu1.Append(wx.ID_ANY, "&Exit")
        self.Bind(wx.EVT_MENU, self.OnExit, exit_id)

        mb.Append(menu1, "&File")

        # Add a 'Help' menu to the menu bar and define its options.
        menu2 = wx.Menu()

        tutorial_id = menu2.Append(wx.ID_ANY, "&Tutorial")
        self.Bind(wx.EVT_MENU, self.OnTutorial, tutorial_id)
        about_id = menu2.Append(wx.ID_ANY, "&About")
        self.Bind(wx.EVT_MENU, self.OnAbout, about_id)

        mb.Append(menu2, "&Help")

        # Attach the menubar to the frame.
        self.SetMenuBar(mb)


    def add_toolbar(self):
        """Create a tool bar and populate it."""

        #tb = self.CreateToolBar()
        tb = wx.ToolBar(parent=self, style=wx.TB_HORIZONTAL|wx.NO_BORDER)

        tb.AddSimpleTool(wx.ID_OPEN, getOpenBitmap(),
                         wx.GetTranslation("Open Data Files"),
                         wx.GetTranslation("Open reflectometry data files"))
        tb.Realize()
        self.SetToolBar(tb)


    def add_statusbar(self, subbars):
        """Create a status bar."""

        sb = self.statusbar = self.CreateStatusBar()
        sb.SetFieldsCount(len(subbars))
        sb.SetStatusWidths(subbars)
        sb.SetStatusText("Welcome to DiRefl", 0)


    def add_notebookbar(self):
        """Create a notebook bar and a set of tabs, one for each page."""

        nb = self.notebook = wx.Notebook(self.panel, wx.ID_ANY,
                                         style=wx.NB_TOP|wx.NB_FIXEDWIDTH)

        # Create page windows as children of the notebook.
        #self.page0 = SimulatedDataPage(nb, colour="", fignum=1)
        self.page0 = SimulatedDataPage(nb, colour="#FFFFB0", fignum=1)  # pale yellow
        #self.page1 = CollectedDataPage(nb, colour="", fignum=0)
        self.page1 = CollectedDataPage(nb, colour="#B0FFB0", fignum=0)  # pale green
        #self.page2 = TestPlotPage(nb, colour="GREEN", fignum=2)
        #self.page3 = TestPlotPage(nb, colour="BLUE", fignum=3)
        #self.page4 = TestPlotPage(nb, colour="YELLOW", fignum=4)
        #self.page5 = TestPlotPage(nb, colour="RED", fignum=5)

        # Add the pages to the notebook with a label to show on the tab.
        nb.AddPage(self.page0, "Simulated Data")
        nb.AddPage(self.page1, "Collected Data")
        #nb.AddPage(self.page2, "Test 1")
        #nb.AddPage(self.page3, "Test 2")
        #nb.AddPage(self.page4, "Test 3")
        #nb.AddPage(self.page5, "Test 4")

        # Put the notebook in a sizer attached to the main panel.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(nb, 1, wx.EXPAND)
        self.panel.SetSizer(sizer)
        sizer.Fit(self.panel)

        '''
        # Sample code to switch windows in notebook tabs
        nb.RemovePage(self.page0)
        nb.RemovePage(self.page1)
        nb.InsertPage(0, self.page1, "Replace 1")
        nb.InsertPage(1, self.page0, "Replace 0")
        '''

    def OnAbout(self, evt):
        """ Show the About dialog. """

        info = wx.AboutDialogInfo()
        info.Name = "DiRefl"
        info.Version = "0.2.0"
        info.Copyright = "(C) 2010 University of Maryland and NIST"
        info.WebSite = ("http://reflectometry.org/danse",
                        "DANSE/Reflectometry home page")
        info.License = wordwrap("BSD License", 100, wx.ClientDC(self))
        info.Description = wordwrap("DiRefl is a reflectometry application \
that computes the scattering length density profile of a thin film or free \
form sample from two experimental (or simulated) reflectometry datasets.  The \
neutron scattering data represent two runs where only one of the surround \
layers (either incident or substrate) was changed and they have sufficient \
contrast via their scattering length densities.  Phase reconstruction and \
direct inversion techniques are used to analyze the data and generate a \
profile of the sample.", 500, wx.ClientDC(self))
        wx.AboutBox(info)


    def OnExit(self, event):
        """Terminate the program."""
        self.Close()


    def OnLoadData(self, event):
        """Load reflectometry data files for measurement 1 and 2."""

        self.page1.col_tab_OnLoadData(event)  # TODO: create menu in dest class


    def OnLoadModel(self, event):
        """Load Model from a file."""

        self.page0.sim_tab_OnLoadModel(event)  # TODO: create menu in dest class


    def OnSaveModel(self, event):
        """Save Model to a file."""

        self.page0.sim_tab_OnSaveModel(event)  # TODO: create menu in dest class


    def OnTutorial(self, event):
        """ Show tutorial information. """

        dlg =wx.MessageDialog(self,
                              message = """\
For a tutorial on how to use DiRefl with sample datasets,\n\
please go to the following webpage:\n\n\
http://www.reflectometry.org/danse/packages.html""",
                              caption = 'Tutorial',
                              style=wx.OK | wx.CENTRE)
        dlg.ShowModal()
        dlg.Destroy()

#==============================================================================

class CollectedDataPage(wx.Panel):
    """
    This class implements phase reconstruction and direct inversion analysis
    of two surround variation data sets (i.e., experimentally collected data)
    to produce a scattering length density profile of the sample.
    """

    def __init__(self, parent, id=wx.ID_ANY, colour="", fignum=0, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.fignum=fignum
        self.SetBackgroundColour("")  #default colour

        # Split the panel to separate the input fields from the plots.
        # wx.SP_LIVE_UPDATE can be omitted to disable repaint as sash is moved.
        sp = wx.SplitterWindow(self, style=wx.SP_3D|wx.SP_LIVE_UPDATE)
        sp.SetMinimumPaneSize(290)

        # Create display panels as children of the splitter.
        self.pan1 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.pan1.SetBackgroundColour(colour)
        self.pan2 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        #self.pan2.SetBackgroundColour("LIGHT GREY")  # same as mpl background
        self.pan2.SetBackgroundColour("WHITE")

        self.init_panel_1()
        self.init_panel_2()

        # Attach the panels to the splitter.
        sp.SplitVertically(self.pan1, self.pan2, sashPosition=300)
        sp.SetSashGravity(0.2)  # on resize grow mostly on right side

        # Put the splitter in a sizer attached to the main page.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sp, 1, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)


    def init_panel_1(self):
        """Initialize the left panel."""

        fields = [ ["SLD of Substrate:", 2.07, "float", None, True],
                   ["SLD of Surface 1:", 6.33, "float", None, True],
                   ["SLD of Surface 2:", 0.0, "float", None, True],
                   ["Sample Thickness:", 1000, "float", None, True],
                   ["Qmin:", 0.0, "float", None, True],
                   ["Qmax:", 0.2, "float", None, True],
                   ["# Profile Steps:", 128, "int", None, True],
                   ["Over Sampling Factor:", 4, "int", None, True],
                   ["# Inversion Iterations:", 6, "int", None, True],
                   ["# Monte Carlo Trials:", 10, "int", None, True],
                ###["Cosine Transform Smoothing:", 0.0, "float", None, True],
                ###["Back Reflectivity:", "True", "str", ("True", "False"), False],
                ###["Inversion Noise Factor:", 1, "int", None, True],
                   ["Bound State Energy:", 0.0, "float", None, True],
                ###["Show Iterations:", "False", "str", ("True", "False"), True]
                ###["Monitor:", "None", "str", None, False]
                 ]

        self.pan3 = ItemListInput(parent=self.pan1, itemlist=fields)

        # Create an introductory section for the panel.
        intro_text = """\
Edit parameters then press Compute button to generate a density profile from \
the data files."""
        intro = wx.TextCtrl(self.pan1, wx.ID_ANY, value=intro_text,
                            style=wx.TE_MULTILINE|wx.TE_WORDWRAP|wx.TE_READONLY)
        intro.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))

        # Create the button controls.
        btn_compute = wx.Button(self.pan1, wx.ID_ANY, "Compute")
        #btn_compute.SetDefault()
        #btn_reset = wx.Button(self.pan1, wx.ID_ANY, "Reset")

        self.Bind(wx.EVT_BUTTON, self.OnCompute, btn_compute)
        #self.Bind(wx.EVT_BUTTON, self.OnReset, btn_reset)

        # Create a horizontal box sizer for the buttons.
        box_a = wx.BoxSizer(wx.HORIZONTAL)
        box_a.Add((10,20), 1)  # stretchable whitespace
        box_a.Add(btn_compute, 0)
        #box_a.Add((10,20), 0)  # non-stretchable whitespace
        #box_a.Add(btn_reset, 0)

        # Create a vertical box sizer for the panel and layout widgets in it.
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(intro, 0, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT, border = 10)
        box.Add(self.pan3, 1, wx.EXPAND|wx.ALL, border = 10)
        box.Add(box_a, 0, wx.EXPAND|wx.BOTTOM|wx.LEFT|wx.RIGHT, 10)

        # Associate the sizer with its container.
        self.pan1.SetSizer(box)
        box.Fit(self.pan1)


    def init_panel_2(self):
        """Initialize the right panel."""

        # Instantiate a figure object that will contain our plots.
        figure = Figure()

        # Initialize the FigureCanvas, mapping the figure object to the plot
        # engine backend.
        canvas = FigureCanvas(self.pan2, wx.ID_ANY, figure)

        # Wx-Pylab magic ...
        # Make our canvas the active figure manager for Pylab so that when
        # pylab plotting statements are executed they will operate on our
        # canvas and not create a new frame and canvas for display purposes.
        # This technique allows this application to execute code that uses
        # pylab stataments to generate plots and embed these plots in our
        # application window(s).
        self.fm = FigureManagerBase(canvas, self.fignum)
        _pylab_helpers.Gcf.set_active(self.fm)

        # Instantiate the matplotlib navigation toolbar and explicitly show it.
        mpl_toolbar = Toolbar(canvas)
        mpl_toolbar.Realize()

        # Create a placeholder for text displayed above the plots.
        intro_text = "Results of phase reconstruction and direct inversion using reflectometry files from two surround measurements."
        intro = wx.StaticText(self.pan2, wx.ID_ANY, label=intro_text)
        intro.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))

        # Create a vertical box sizer for the panel to layout its widgets.
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(intro, 0, wx.EXPAND|wx.ALL, border = 10)
        box.Add(canvas, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)
        box.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=10)

        # Associate the sizer with its container.
        self.pan2.SetSizer(box)
        box.Fit(self.pan2)

        curdir = os.path.dirname(os.path.realpath(__file__))
        self.data_file_1 = os.path.join(curdir, "qrd1.refl")
        self.data_file_2 = os.path.join(curdir, "qrd2.refl")

        #self.pan2.Bind(wx.EVT_MOTION, self.OnPan2Motion)


    def OnPan2Motion(self, event):
        """ Display cursor position in status bar."""

        write_to_statusbar("%s" % str(event.GetPositionTuple()), 4)


    def OnCompute(self, event):
        """Execute the operation."""

        import pylab
        import time

        # Explicitly validate all input parameters before proceeding.  The
        # panel's Validate method will invoke all validators associated with
        # its top-level input objects and transfer data from them.  Although
        # char-by-char validation would have warned the user about any invalid
        # entries, the user could have pressed the Compute button without
        # making the corrections, so a full validation pass must be done now.
        if not self.pan3.Validate():
            display_error_message(self, "Data Entry Error",
                "Please correct the highlighted fields in error.")
            return

        self.args = [self.data_file_1, self.data_file_2]

        # Get the validated parameters.
        self.params = self.pan3.GetResults()
        #print "Results from %d input fields:" %(len(self.params))
        #print "  ", self.params

        # Inform the user that we're entering a period of high computation.
        write_to_statusbar("Generating new plots ...", 2)
        write_to_statusbar("", 3)

        # Set the plotting figure manager for this class as the active one and
        # erase the current figure.
        _pylab_helpers.Gcf.set_active(self.fm)
        pylab.clf()
        pylab.draw()

        # Perform the phase inversion and direct inversion using new parameters.
        t0 = time.time()
        perform_recon_inver(self.args, self.params)
        pylab.draw()
        secs = time.time() - t0

        # Write the total execution and plotting time to the status bar.
        write_to_statusbar("Plots updated", 2)
        write_to_statusbar("%g secs" %(secs), 3)


    def OnReset(self, event):
        pass


    def col_tab_OnLoadData(self, event):
    #def OnLoadData(self, event):  # TODO: reorganize menu to call directly
        """Load reflectometry data files for measurement 1 and 2."""

        dlg = wx.FileDialog(self,
                            message="Load Data for Measurement 1",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=REFL_FILES+"|"+TEXT_FILES+"|"+ALL_FILES,
                            style=wx.OPEN)
        # Wait for user to close the dialog.
        sts = dlg.ShowModal()
        if sts == wx.ID_OK:
            pathname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            filespec = os.path.join(pathname, filename)
        dlg.Destroy()
        if sts == wx.ID_CANCEL:
            return  # Do nothing

        self.data_file_1 = filespec

        dlg = wx.FileDialog(self,
                            message="Load Data for Measurement 2",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=REFL_FILES+"|"+TEXT_FILES+"|"+ALL_FILES,
                            style=wx.OPEN)
        # Wait for user to close the dialog.
        sts = dlg.ShowModal()
        if sts == wx.ID_OK:
            pathname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            filespec = os.path.join(pathname, filename)
        dlg.Destroy()
        if sts == wx.ID_CANCEL:
            return  # Do nothing

        self.data_file_2 = filespec

#==============================================================================

class SimulatedDataPage(wx.Panel):
    """
    This class implements phase reconstruction and direct inversion analysis
    of two simulated surround variation data sets (generated from a model)
    to produce a scattering length density profile of the sample.
    """

    def __init__(self, parent, id=wx.ID_ANY, colour="", fignum=0, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.fignum=fignum
        self.SetBackgroundColour(colour)

        # Split the panel to separate the input fields from the plots.
        # wx.SP_LIVE_UPDATE can be omitted to disable repaint as sash is moved.
        sp = wx.SplitterWindow(self, style=wx.SP_3D|wx.SP_LIVE_UPDATE)
        sp.SetMinimumPaneSize(290)

        # Create display panels as children of the splitter.
        self.pan1 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.pan1.SetBackgroundColour("")  # default colour
        self.pan2 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        #self.pan2.SetBackgroundColour("LIGHT GREY")  # same as mpl background
        self.pan2.SetBackgroundColour("WHITE")

        self.init_panel_1()
        self.init_panel_2()

        # Attach the panels to the splitter.
        sp.SplitVertically(self.pan1, self.pan2, sashPosition=300)
        sp.SetSashGravity(0.2)  # on resize grow mostly on right side

        # Put the splitter in a sizer attached to the main page.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sp, 1, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)


    def init_panel_1(self):
        """Initialize the left panel."""

        fields = [
                ###["SLD of Substrate:", 2.07, "float", None, True],
                   ["SLD of Surface 1:", 0.0, "float", None, True],
                   ["SLD of Surface 2:", 4.5, "float", None, True],
                ###["Sample Thickness:", 1000, "float", None, True],
                   ["Qmin:", 0.0, "float", None, True],
                   ["Qmax:", 0.4, "float", None, True],
                   ["# Profile Steps:", 128, "int", None, True],
                   ["Over Sampling Factor:", 4, "int", None, True],
                   ["# Inversion Iterations:", 6, "int", None, True],
                   ["# Monte Carlo Trials:", 10, "int", None, True],
                ###["Cosine Transform Smoothing:", 0.0, "float", None, True],
                ###["Back Reflectivity:", "True", "str", ("True", "False"), True],
                ###["Inversion Noise Factor:", 1, "int", None, True],
                   ["Simulated Noise (as %):", 8.0, "float", None, True],
                   ["Bound State Energy:", 0.0, "float", None, True],
                   ["Perfect Reconstruction:", "False", "str", ("True", "False"), True],
                ###["Show Iterations:", "False", "str", ("True", "False"), True]
                ###["Monitor:", "None", "str", None, False]
                 ]

        self.pan3 = ItemListInput(parent=self.pan1, itemlist=fields)

        # Create an introductory section for the panel.
        intro_text = """\
Edit parameters then press Compute button to generate a density profile from \
your model."""
        intro = wx.TextCtrl(self.pan1, wx.ID_ANY, value=intro_text,
                            style=wx.TE_MULTILINE|wx.TE_WORDWRAP|wx.TE_READONLY)
        intro.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))

        sbox = wx.StaticBox(self.pan1, wx.ID_ANY, "Model Parameters")
        sbox_sizer = wx.StaticBoxSizer(sbox, wx.VERTICAL)

        stxt1 = wx.StaticText(self, wx.ID_ANY,
                    label="Define the Surface, Sample Layers, and Substrate")
        stxt2 = wx.StaticText(self, wx.ID_ANY,
                    label="    components of your model (one layer per line):")
        #stxt3 = wx.StaticText(self, wx.ID_ANY,
        #           label="    as shown below (roughness defaults to 0):")

        # Read in demo model parameters.
        # Note that the number of lines determines the height of the box.
        # TODO: create a model edit box with a min-max height.
        demoname = "demo_model_1.dat"
        curdir = os.path.dirname(os.path.realpath(__file__))
        filespec = os.path.join(curdir, demoname)

        try:
            fd = open(filespec, 'rU')
            demo_model_params = fd.read()
            fd.close()
        except:
            display_warning_message(self, "Load Model Error",
                "Error loading demo model from file "+demoname)
            demo_model_params = \
                "# SLDensity  Thickness  Roughness  comments\n\n\n\n\n\n\n"

        # Create an input box to enter and edit the model description and
        # populate it with the contents of the demo file.
        self.model = wx.TextCtrl(self.pan1, wx.ID_ANY,
                                 value=demo_model_params,
                                 style=wx.TE_MULTILINE|wx.TE_WORDWRAP)
        #self.model.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))

        sbox_sizer.Add(stxt1, 0, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT, border=10)
        sbox_sizer.Add(stxt2, 0, wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)
        #sbox_sizer.Add(stxt3, 0, wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)
        sbox_sizer.Add(self.model, 1, wx.EXPAND|wx.BOTTOM|wx.LEFT|wx.RIGHT, border=10)

        # Create the button controls.
        btn_compute = wx.Button(self.pan1, wx.ID_ANY, "Compute")
        #btn_compute.SetDefault()
        #btn_reset = wx.Button(self.pan1, wx.ID_ANY, "Reset")

        self.Bind(wx.EVT_BUTTON, self.OnCompute, btn_compute)
        #self.Bind(wx.EVT_BUTTON, self.OnReset, btn_reset)

        # Create a horizontal box sizer for the buttons.
        box_a = wx.BoxSizer(wx.HORIZONTAL)
        box_a.Add((10,20), 1)  # stretchable whitespace
        box_a.Add(btn_compute, 0)
        #box_a.Add((10,20), 0)  # non-stretchable whitespace
        #box_a.Add(btn_reset, 0)

        # Create a vertical box sizer for the panel and layout widgets in it.
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(intro, 0, wx.EXPAND|wx.ALL, border = 10)
        box.Add(sbox_sizer, 0, wx.EXPAND|wx.LEFT|wx.RIGHT, border = 10)
        box.Add(self.pan3, 1, wx.EXPAND|wx.ALL, border = 10)
        box.Add(box_a, 0, wx.EXPAND|wx.BOTTOM|wx.LEFT|wx.RIGHT, 10)

        # Associate the sizer with its container.
        self.pan1.SetSizer(box)
        box.Fit(self.pan1)


    def init_panel_2(self):
        """Initialize the right panel."""

        # Instantiate a figure object that will contain our plots.
        figure = Figure()

        # Initialize the FigureCanvas, mapping the figure object to the plot
        # engine backend.
        canvas = FigureCanvas(self.pan2, wx.ID_ANY, figure)

        # Wx-Pylab magic ...
        # Make our canvas the active figure manager for Pylab so that when
        # pylab plotting statements are executed they will operate on our
        # canvas and not create a new frame and canvas for display purposes.
        # This technique allows this application to execute code that uses
        # pylab stataments to generate plots and embed these plots in our
        # application window(s).
        self.fm = FigureManagerBase(canvas, self.fignum)
        _pylab_helpers.Gcf.set_active(self.fm)

        # Instantiate the matplotlib navigation toolbar and explicitly show it.
        mpl_toolbar = Toolbar(canvas)
        mpl_toolbar.Realize()

        # Create a placeholder for text displayed above the plots.
        intro_text = "Results of phase reconstruction and direct inversion using simulated data files generated from model parameters"
        intro = wx.StaticText(self.pan2, wx.ID_ANY, label=intro_text)
        intro.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))

        # Create a vertical box sizer for the panel to layout its widgets.
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(intro, 0, wx.EXPAND|wx.ALL, border = 10)
        box.Add(canvas, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)
        box.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=10)

        # Associate the sizer with its container.
        self.pan2.SetSizer(box)
        box.Fit(self.pan2)


    def OnCompute(self, event):
        """Execute the operation."""

        import pylab
        import time

        # Explicitly validate all input parameters before proceeding.  The
        # panel's Validate method will invoke all validators associated with
        # its top-level input objects and transfer data from them.  Although
        # char-by-char validation would have warned the user about any invalid
        # entries, the user could have pressed the Compute button without
        # making the corrections, so a full validation pass must be done now.
        if not self.pan3.Validate():
            display_error_message(self, "Data Entry Error",
                "Please correct the highlighted fields in error.")
            return

        # Get the validated parameters.
        self.params = self.pan3.GetResults()

        # Validate and convert the model description into a list of layers.
        lines = self.model.GetValue().splitlines()
        layers = []
        for line in lines:
            lin = line.strip()
            if lin.startswith('#'): continue  # skip over comment line
            if len(lin) == 0: continue  # discard blank line
            keep = lin.split('#')
            lin = keep[0]  # discard trailing comment
            ln = lin.split(None, 4)  # we'll break into a max of 4 items
            if len(ln) == 1: ln.append('100')  # default thickness to 100
            if len(ln) == 2: ln.append('0')  # default roughness to 0.0

            try:
                temp = [ float(ln[0]), float(ln[1]), float(ln[2]) ]
            except:
                display_error_message(self, "Syntax Error",
                    "Please correct syntax error in model description.")
                return

            layers.append(temp)

        if len(layers) < 3:
            display_error_message(self, "Less Than 3 Layer Defined",
                "You must specify at least one Surface, Sample, and Substrate layer.")
            return

        sample = layers[1:-1]
        #print "=== layers", layers
        #print "=== sample", sample
        self.params.append(layers[-1][0])  # add SLD of substrate to list
        self.params.append(layers[-1][2])  # add roughness of substrate to list

        # Inform the user that we're entering a period of high computation.
        write_to_statusbar("Generating new plots ...", 2)
        write_to_statusbar("", 3)

        # Set the plotting figure manager for this class as the active one and
        # erase the current figure.
        _pylab_helpers.Gcf.set_active(self.fm)
        pylab.clf()
        pylab.draw()

        # Perform the phase inversion and direct inversion using new parameters.
        t0 = time.time()
        perform_simulation(sample, self.params)
        pylab.draw()
        secs = time.time() - t0

        # Write the total execution and plotting time to the status bar.
        write_to_statusbar("Plots updated", 2)
        write_to_statusbar("%g secs" %(secs), 3)


    def OnReset(self, event):
        pass


    def sim_tab_OnLoadModel(self, event):
    #def OnLoadModel(self, event):  # TODO: reorganize menu to call directly
        """Load Model from a file."""

        dlg = wx.FileDialog(self,
                            message="Load Model from File ...",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=DATA_FILES+"|"+TEXT_FILES+"|"+ALL_FILES,
                            style=wx.OPEN)
        # Wait for user to close the dialog.
        sts = dlg.ShowModal()
        if sts == wx.ID_OK:
            pathname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            filespec = os.path.join(pathname, filename)
        dlg.Destroy()
        if sts == wx.ID_CANCEL:
            return  # Do nothing

        # Read the entire input file into a buffer.
        try:
            fd = open(filespec, 'rU')
            model_params = fd.read()
            fd.close()
        except:
            display_error_message(self, "Load Model Error",
                                  "Error loading model from file "+filename)
            return

        # Replace the contents of the model parameter text control box with
        # the data from the file.
        self.model.Clear()
        self.model.SetValue(model_params)


    def sim_tab_OnSaveModel(self, event):
    #def OnSaveModel(self, event):  # TODO: reorganize menu to call directly
        """Save Model to a file."""

        dlg = wx.FileDialog(self,
                            message="Save Model to File ...",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=DATA_FILES+"|"+TEXT_FILES+"|"+ALL_FILES,
                            style=wx.SAVE|wx.OVERWRITE_PROMPT)
        # Wait for user to close the dialog.
        sts = dlg.ShowModal()
        if sts == wx.ID_OK:
            pathname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            filespec = os.path.join(pathname, filename)
        dlg.Destroy()
        if sts == wx.ID_CANCEL:
            return  # Do nothing

        # Put the contents of the model parameter text control box into a
        # buffer.
        model_params = self.model.GetValue()

        # Write the entire buffer to the output file.
        try:
            fd = open(filespec, 'w')
            fd.write(model_params)
            fd.close()
        except:
            display_error_message(self, "Save Model Error",
                                  "Error saving model to file "+filename)
            return

#==============================================================================

class TestPlotPage(wx.Panel):
    """This class implements a page of the notebook."""

    def __init__(self, parent, id=wx.ID_ANY, colour="", fignum=1, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.fignum=fignum
        self.SetBackgroundColour(colour)

        self.pan1 = wx.Panel(self, wx.ID_ANY, style=wx.SUNKEN_BORDER)

        self.init_panel_1()

        # Put the panel in a sizer attached to its parent panel.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.pan1, 1, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)


    def init_panel_1(self):
        """Initialize the first panel."""

        # Instantiate a figure object that will contain our plots.
        figure = Figure()

        # Initialize the FigureCanvas, mapping the figure object to the plot
        # engine backend.
        canvas = FigureCanvas(self.pan1, wx.ID_ANY, figure)

        # Wx-Pylab magic ...
        # Make our canvas the active figure manager for Pylab so that when
        # pylab plotting statements are executed they will operate on our
        # canvas and not create a new frame and canvas for display purposes.
        # This technique allows this application to execute code that uses
        # pylab stataments to generate plots and embed these plots in our
        # application window(s).
        fm = FigureManagerBase(canvas, self.fignum)
        _pylab_helpers.Gcf.set_active(fm)

        # Instantiate the matplotlib navigation toolbar and explicitly show it.
        mpl_toolbar = Toolbar(canvas)
        mpl_toolbar.Realize()

        # Create a vertical box sizer for the panel to layout its widgets.
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(canvas, 1, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT, border=10)
        box.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=10)

        # Associate the sizer with its container.
        self.pan1.SetSizer(box)
        box.Fit(self.pan1)

        if self.fignum==4: test3()
        if self.fignum==5: test4(figure)
        import sys
        if len(sys.argv) < 2: return
        if (self.fignum==2 and '-test1' in sys.argv): test1()
        if (self.fignum==3 and '-test2' in sys.argv): test2()

#==============================================================================

class InversionApp(wx.App):
    """This class implements the main application window."""

    def OnInit(self):
        # Compute the size of the application frame such that it fits on the
        # user's screen without obstructing (or being obstructed by) the
        # Windows launch bar and has a miximum initial size of DISPLAY_WIDTHxDISPLAY_HEIGTH pixels.
        xpos = ypos = 0
        x, y = wx.DisplaySize()
        y -= TASKBAR_HEIGHT  # avoid obscuring the Windows task bar
        if x > DISPLAY_WIDTH : xpos = (x - DISPLAY_WIDTH)/2
        if y > DISPLAY_HEIGHT : ypos = (y - DISPLAY_HEIGHT)/2

        frame = AppFrame(title="Phase Reconstruction and Direct Inversion Reflectometry",
                         pos=(xpos, ypos),
                         size=(min(x, DISPLAY_WIDTH), min(y, DISPLAY_HEIGHT)))
        frame.Show(True)
        self.SetTopWindow(frame)
        return True

#==============================================================================

def write_to_statusbar(text, index):
    """Write a message to the status bar in the specified slot."""

    frame = wx.FindWindowByName("AppFrame", parent=None)
    frame.statusbar.SetStatusText(text, index)


def display_error_message(win, title, msg):
    """Display an error message in a pop-up dialog box with an OK button."""

    # Display message with padding at end for better appearance.
    msg = wx.MessageDialog(win, msg+'     ', title, wx.ICON_ERROR|wx.OK)
    msg.ShowModal()
    msg.Destroy()


def display_warning_message(win, title, msg):
    """Display a warning message in a pop-up dialog box with an OK button."""

    # Display message with padding at end for better appearance.
    msg = wx.MessageDialog(win, msg+'     ', title, wx.ICON_WARNING|wx.OK)
    msg.ShowModal()
    msg.Destroy()


def perform_recon_inver(args, params):
    """
    Perform phase reconstruction and direct inversion on two reflectometry data
    sets to generation a scattering length depth profile of the sample.
    """

    from core import refl, SurroundVariation, Inversion
    import os
    import pylab

    u = params[0]
    v1 = params[1]
    v2 = params[2]
    phase = SurroundVariation(args[0], args[1], u=u, v1=v1, v2=v2)
    data = phase.Q, phase.RealR, phase.dRealR

    #if dz: rhopoints = ceil(1/dz)
    #_backrefl = True if params[99] == "True" else False
    _backrefl = True
    #_showiters = True if params[99] == "True" else False
    _showiters = False

    res = Inversion(data=data, **dict(substrate=u,
                                      thickness=params[3],
                                      Qmin=params[4],
                                      Qmax=params[5],
                                      #Qmax=None,
                                      rhopoints=params[6],
                                      calcpoints=params[7],
                                      iters=params[8],
                                      stages=params[9],
                                      ctf_window=0, #cosine transform smoothing
                                      backrefl=_backrefl,
                                      noise=1,  # inversion noise factor
                                      bse=params[10],
                                      showiters=_showiters,
                                      monitor=None))
    res.run(showiters=False)
    res.plot(phase=phase)

    pylab.subplots_adjust(wspace=0.25, hspace=0.33,
                          left=0.09, right=0.96,
                          top=0.95, bottom=0.08)


def perform_simulation(sample, params):
    """
    Simulate reflectometry data sets from model information then perform
    phase reconstruction and direct inversion on the data to generate a
    scattering length density profile.
    """

    from simulate import Simulation
    from numpy import linspace
    import pylab


    if sample is None:
        # Roughness parameters (surface, sample, substrate)
        sv, s, su = 3, 5, 2
        # Surround parameters
        u, v1, v2 = 2.07, 0, 4.5
        # Default sample
        sample = ([5,100,s], [1,123,s], [3,47,s], [-1,25,s])
        sample[0][2] = sv
    else:
        su = 2

    # Run the simulation
    _perfect_reconstruction = True if params[10] == "True" else False
    #_showiters = True if params[99] == "True" else False
    _showiters = False
    _noise = params[8]
    if _noise < 0.01: _noise = 0.01
    _noise /= 100.0  # convert percent value to hundreths value

    inv = dict(showiters=_showiters,
               monitor=None,
               bse=params[9],
               noise=1,  # inversion noise factor
               iters=params[6],
               stages=params[7],
               rhopoints=params[4],
               calcpoints=params[5])
    t = Simulation(q=linspace(params[2], params[3], 150),
                   sample=sample,
                   u=params[11],
                   urough=params[12],
                   v1=params[0],
                   v2=params[1],
                   noise=_noise,
                   invert_args=inv,
                   phase_args=dict(stages=100),
                   perfect_reconstruction=_perfect_reconstruction)
    t.plot()
    pylab.subplots_adjust(wspace=0.25, hspace=0.33,
                          left=0.09, right = 0.96,
                          top=0.95, bottom=0.08)


def test1():
    """
    Test interface to phase reconstruction and direct inversion routines
    in core.py using two actual reflectometry data files.
    """

    from core import refl, SurroundVariation, Inversion
    import os
    import pylab

    #args = ['wsh02_re.dat']
    curdir = os.path.dirname(os.path.realpath(__file__))
    args = [os.path.join(curdir, "qrd1.refl"),
            os.path.join(curdir, "qrd2.refl")]
    if len(args) == 1:
        phase = None
        data = args[0]
    elif len(args) == 2:
        v1 = 6.33
        v2 = 0.0
        u = 2.07
        phase = SurroundVariation(args[0], args[1], u=u, v1=v1, v2=v2)
        data = phase.Q, phase.RealR, phase.dRealR

    #if dz: rhopoints = ceil(1/dz)
    res = Inversion(data=data, **dict(substrate=2.07,
                                      thickness=1000,
                                      calcpoints=4,
                                      rhopoints=128,
                                      Qmin=0,
                                      Qmax=None,
                                      iters=6,
                                      stages=10,
                                      ctf_window=0,
                                      backrefl=True,
                                      noise=1,
                                      bse=0,
                                      showiters=False,
                                      monitor=None))
    res.run(showiters=False)
    res.plot(phase=phase)

    pylab.subplots_adjust(wspace=0.25, hspace=0.33,
                          left=0.09, right=0.96,
                          top=0.95, bottom=0.08)


def test2():
    """
    Test interface to simulation routine in simulation.py using a reconstructed
    reflectometry data file.
    """

    from simulate import Simulation
    from numpy import linspace
    import pylab

    # Roughness parameters (surface, sample, substrate)
    sv, s, su = 3, 5, 2
    # Surround parameters
    u, v1, v2 = 2.07, 0, 4.5
    # Default sample
    sample = ([5,100,s], [1,123,s], [3,47,s], [-1,25,s])
    sample[0][2] = sv
    bse = 0

    # Run the simulation
    inv = dict(showiters=False, iters=6, monitor=None, bse=bse,
               noise=1, stages=10, calcpoints=4, rhopoints=128)
    t = Simulation(q = linspace(0, 0.4, 150), sample=sample,
                   u=u, urough=su, v1=v1, v2=v2, noise=0.08,
                   invert_args=inv, phase_args=dict(stages=100),
                   perfect_reconstruction=False)
    t.plot()
    pylab.subplots_adjust(wspace=0.25, hspace=0.33,
                          left=0.09, right = 0.96,
                          top=0.95, bottom=0.08)


def test3():
    """
    Test the ability to utilize code that uses the procedural interface
    to pylab to generate subplots.
    """

    import pylab

    pylab.suptitle("Test use of procedural interface to Pylab", fontsize=16)

    pylab.subplot(211)
    x = np.arange(0, 6, .01)
    y = np.sin(x**2)*np.exp(-x)
    pylab.xlabel("x-axis")
    pylab.ylabel("y-axis")
    pylab.title("First Plot")
    pylab.plot(x, y)

    pylab.subplot(212)
    x = np.arange(0, 8, .01)
    y = np.sin(x**2)*np.exp(-x) + 1
    pylab.xlabel("x-axis")
    pylab.ylabel("y-axis")
    pylab.title("Second Plot")
    pylab.plot(x, y)

    #pylab.show()


def test4(figure):
    """
    Test the ability to utilize code that uses the object oriented interface
    to pylab to generate subplots.
    """

    import pylab

    axes = figure.add_subplot(311)
    x = np.arange(0, 6, .01)
    y = np.sin(x**2)*np.exp(-x)
    axes.plot(x, y)

    axes = figure.add_subplot(312)
    x = np.arange(0, 8, .01)
    y = np.sin(x**2)*np.exp(-x) + 1
    axes.plot(x, y)
    axes.set_ylabel("y-axis")

    axes = figure.add_subplot(313)
    x = np.arange(0, 4, .01)
    y = np.sin(x**2)*np.exp(-x) + 2
    axes.plot(x, y)
    axes.set_xlabel("x-axis")

    pylab.suptitle("Test use of object oriented interface to Pylab",
                   fontsize=16)
    pylab.subplots_adjust(hspace=0.35)
    #pylab.show()

#==============================================================================

if __name__ == '__main__':
    # Instantiate the application class and give control to wxPython.
    app = InversionApp(redirect=False, filename=None)
    app.MainLoop()
