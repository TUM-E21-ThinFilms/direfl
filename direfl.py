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

"""
This module is used to start the GUI for the Direct Inversion Reflectometry
application.  It creates the initial wxPython frame, presents a splash screen
to the user, and then constructs the rest of the GUI.

From the command line, the application can be run as follows:

$ python direfl.py [<optional parameters>]

The following is a list of command line parameters for development and
debugging purposes.  None are documented and they may change at any time.

Options for showing diagnostic info:
    --platform      Display platform specific info, especially about fonts
    --tracep        Display values from user input fields
    --debug         Display info for debugging purposes (changes frequently)
    --time          Display diagnostic timing information
    --syspath       Display the contents of sys.path

Options for overriding the default font and point size attributes where
parameters within each set are mutually exclusive (last one takes precedence):
    --tahoma, --arial, --verdana
    --6pt, --7pt, --8pt, --9pt, --10pt, --11pt, --12pt

Options for controlling the development and testing environment:
    --xtabs         Add extra notebook pages for test purposes
        --test1     Execute test1() in a test page
        --test2     Execute test2() in a test page
    --inspect       Run the wxPython Widget Inspection Tool in a debug window
"""

#==============================================================================

import os
import sys
import time

import wx

from common.utilities import get_appdir, log_time


if len(sys.argv) > 1 and '--syspath' in sys.argv[1:]:
    print "*** App root dir is", get_appdir(),
    print "*** __file__ is" , __file__
    print "*** Python path is:"
    for i, p in enumerate(sys.path):
        print "%5d  %s" %(i, p)

from gui.app_frame import AppFrame
from gui.about import APP_TITLE

# Desired initial window size (if physical screen size permits).
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 900

# Resource files.
PROG_SPLASH_SCREEN = "splash.png"

#==============================================================================

class DiReflApp(wx.App):
    """
    This class implements the wxPyton based GUI for the Direct Inversion
    Reflectometry application.
    """

    # Design note: The basic application frame is created, the splash screen is
    # displayed, and finally the frame is populated.  The frame initialization
    # is performed in two parts so that the splash screen is visible while the
    # application specific packages are imported and the application is set up.

    def OnInit(self):
        # Determine the position and size of the application frame and likewise
        # for the splash window that will cover it.
        pos, size = self.window_placement()
        #print "window pos and size =", pos, size

        # Create a basic application frame without any child panels.
        frame = AppFrame(parent=None, title=APP_TITLE, pos=pos, size=size)

        # Display a splash screen on top of the frame.
        if len(sys.argv) > 1 and '--time' in sys.argv[1:]:
            log_time("Starting to display the splash screen")
        self.display_splash_screen(frame)

        # Create the graphical user interface for the application on the frame.
        if len(sys.argv) > 1 and '--time' in sys.argv[1:]:
            log_time("Starting to build the GUI on the frame")
        frame.init_GUI()

        frame.Show(True)
        self.SetTopWindow(frame)

        # The splash screen can be dismissed by the user via a left mouse click
        # as soon as the wxPython event loop is entered (i.e. when the caller
        # executes app.MainLoop()).  Otherwise, the splash screen will stop
        # itself when its timeout expires.
        return True


    def window_placement(self):
        """
        Determines the position and size of the application frame such that it
        fits on the user's screen without obstructing (or being obstructed by)
        the Windows task bar.  The maximum initial size in pixels is bounded by
        DISPLAY_WIDTH x DISPLAY_HEIGHT.  For most monitors, the application
        will be centered on the screen; for very large monitors it will be
        placed on the left side of the screen.
        """

        xpos = ypos = 0

        # Note that when running Linux and using an Xming (X11) server on a PC
        # with a dual  monitor configuration, the reported display size may be
        # that of both monitors combined with an incorrect display count of 1.
        # To avoid displaying this app across both monitors, we check for
        # screen 'too big'.  If so, we assume a smaller width which means the
        # application will be placed towards the left hand side of the screen.

        j, k, x, y = wx.Display().GetClientArea() # size excludes task bar
        if len(sys.argv) > 1 and '--platform' in sys.argv[1:]:
            w, h = wx.DisplaySize()  # size includes task bar area
            print "*** Reported screen size including taskbar is %d x %d"%(w, h)
            print "*** Reported screen size excluding taskbar is %d x %d"%(x, y)

        if x > 1920: x = 1280  # display on left side, not centered on screen
        if x > DISPLAY_WIDTH:  xpos = (x - DISPLAY_WIDTH)/2
        if y > DISPLAY_HEIGHT: ypos = (y - DISPLAY_HEIGHT)/2

        # Return the suggested position and size for the application frame.
        return (xpos, ypos), (min(x, DISPLAY_WIDTH), min(y, DISPLAY_HEIGHT))


    def display_splash_screen(self, parent):
        """Displays the splash screen.  It will exactly cover the main frame."""

        # Prepare the picture.  On a 2GHz intel cpu, this takes about a second.
        x, y = parent.GetSizeTuple()
        image = wx.Image(os.path.join(get_appdir(), PROG_SPLASH_SCREEN),
                         wx.BITMAP_TYPE_PNG)
        image.Rescale(x, y, wx.IMAGE_QUALITY_HIGH)
        bm = image.ConvertToBitmap()

        # Create and show the splash screen.  It will disappear only when the
        # program has entered the event loop AND either the timeout has expired
        # or the user has left clicked on the screen.  Thus any processing
        # performed in this routine (including sleeping) or processing in the
        # calling routine (including doing imports) will prevent the splash
        # screen from disappearing.
        #
        # Note that on Linux, the timeout appears to occur immediately in which
        # case the splash screen disappears upon entering the event loop.
        wx.SplashScreen(bitmap=bm,
                        splashStyle=(wx.SPLASH_CENTRE_ON_PARENT|
                                     wx.SPLASH_TIMEOUT|
                                     wx.STAY_ON_TOP),
                        milliseconds=4000,
                        parent=parent,
                        id=wx.ID_ANY)

        # Keep the splash screen up a minimum amount of time for non-Windows
        # systems.  This is a workaround for Linux and possibly MacOS that
        # appear to ignore the splash screen timeout option.
        if '__WXMSW__' not in wx.PlatformInfo:
            if len(sys.argv) > 1 and '--time' in sys.argv[1:]:
                log_time("Starting sleep of 2 secs")
            time.sleep(2)

        # A call to wx.Yield does not appear to be required.  If used on
        # Windows, the cursor changes from 'busy' to 'ready' before the event
        # loop is reached which is not desirable.  On Linux it seems to have
        # no effect.
        #wx.Yield()

#==============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1 and '--time' in sys.argv[1:]:
        log_time("Starting DiRefl")

    # Instantiate the application class and give control to wxPython.
    app = DiReflApp(redirect=False, filename=None)

    # For wx debugging, load the wxPython Widget Inspection Tool if requested.
    # It will cause a separate interactive debugger window to be displayed.
    if len(sys.argv) > 1 and '--inspect' in sys.argv[1:]:
        import wx.lib.inspection
        wx.lib.inspection.InspectionTool().Show()

    if len(sys.argv) > 1 and '--time' in sys.argv[1:]:
        log_time("Done initializing - entering the event loop")

    app.MainLoop()
