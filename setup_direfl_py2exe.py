#!/usr/bin/env python

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

# Author: James Krycka

"""
This script uses py2exe to create inversion\dist\direfl.exe.

The resulting executable bundles the DiRefl application, the python runtime
environment, and other required python packages into a single file.  Additional
resource files that are needed when DiRefl is run are placed in the dist
directory tree.  On completion, the contents of the dist directory tree can be
used by the Inno Setup Compiler (via a separate script) to build a Windows
installer/uninstaller for deployment of the DiRefl application.  For testing
purposes, direfl.exe can be run from the dist directory.
"""

import os
import sys

'''
print "*** Python path is:"
for i, p in enumerate(sys.path):
    print "%5d  %s" %(i, p)
'''

from distutils.core import setup

# Augment the setup interface with the py2exe command and make sure the py2exe
# option is passed to setup.
import py2exe

if len(sys.argv) == 1:
    sys.argv.append('py2exe')

import matplotlib

# Retrieve the application version string.
from version import version as version

# Create a manifest for use with Python 2.5 on Windows XP.  This manifest is
# required to be included in a py2exe image (or accessible as a file in the
# image directory) when wxPython is included so that the Windows XP theme is
# used when rendering wx widgets.  The manifest below is adapted from the
# Python manifest file (C:\Python25\pythonw.exe.manifest).
#
# Note that a different manifest is required if using another version of Python.

manifest = """
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
<assemblyIdentity
    version="0.64.1.0"
    processorArchitecture="x86"
    name="Controls"
    type="win32"
/>
<description>DiRefl</description>
<dependency>
    <dependentAssembly>
        <assemblyIdentity
            type="win32"
            name="Microsoft.Windows.Common-Controls"
            version="6.0.0.0"
            processorArchitecture="X86"
            publicKeyToken="6595b64144ccf1df"
            language="*"
        />
    </dependentAssembly>
</dependency>
</assembly>
"""

# Create a list of all files to include along side the executable being built
# in the dist directory tree.  Each element of the data_files list is a tuple
# consisting of a path (relative to dist\) and a list of files in that path.
data_files = []

# Add data files from the matplotlib\mpl-data folder and its subfolders.
# For matploblib prior to version 0.99 see the examples at the end of the file.
data_files = matplotlib.get_py2exe_datafiles()

# Add resource files that need to reside in the same directory as the image.
data_files.append( ('.', [os.path.join('.', 'demo_model_1.dat')]) )
data_files.append( ('.', [os.path.join('.', 'demo_model_2.dat')]) )
data_files.append( ('.', [os.path.join('.', 'demo_model_3.dat')]) )
data_files.append( ('.', [os.path.join('.', 'direfl.ico')]) )
data_files.append( ('.', [os.path.join('.', 'LICENSE-direfl.txt')]) )
data_files.append( ('.', [os.path.join('.', 'qrd1.refl')]) )
data_files.append( ('.', [os.path.join('.', 'qrd2.refl')]) )
data_files.append( ('.', [os.path.join('.', 'surround_air_4.refl')]) )
data_files.append( ('.', [os.path.join('.', 'surround_d2o_4.refl')]) )
data_files.append( ('.', [os.path.join('.', 'README-direfl.txt')]) )
data_files.append( ('.', [os.path.join('.', 'splash.png')]) )

# Specify required packages to bundle in the executable image.
packages = ['matplotlib', 'numpy', 'scipy', 'pytz']

# Specify files to include in the executable image.
includes = []

# Specify files to exclude from the executable image.
excludes = ['Tkinter', 'PyQt4']

dll_excludes = ['MSVCR71.dll',
                'w9xpopen.exe',
                'libgdk_pixbuf-2.0-0.dll',
                'libgobject-2.0-0.dll',
                'libgdk-win32-2.0-0.dll',
                'cygwin1.dll',
                'tcl84.dll',
                'tk84.dll',
                'QtGui4.dll',
                'QtCore4.dll']

class Target():
    """This class stores metadata about the distribution in a dictionary."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.version = version
        self.company_name = "University of Maryland"
        self.copyright = "BSD style copyright"

client = Target(
    name = 'FOO DiRefl',
    description = 'FOO Direct Inversion and Phase Reconstruction',
    script = 'direfl.py',  # module to run on application start
    dest_base = 'direfl',  # file name part of the exe file to create
    icon_resources = [(1, 'direfl.ico')],  # also need to specify in data_files
    bitmap_resources = [],
    other_resources = [(24, 1, manifest)])

# Now do the work to create a standalone distribution using py2exe.
# Specify either console mode or windows mode build.
#
# When the application is run in console mode, a console window will be created
# to receive any logging or error messages and the application will then create
# a separate GUI application window.
#
# When the application is run in windows mode, it will create a GUI application
# window and no console window will be provided.
setup(
      #console=[client],
      windows=[client],
      options={'py2exe': {
                   'packages': packages,
                   'includes': includes,
                   'excludes': excludes,
                   'dll_excludes': dll_excludes,
                   'compressed': 1,   # standard compression
                   'optimize': 0,     # no byte-code optimization
                   'dist_dir': "dist",# where to put py2exe results
                   'xref': False,     # display cross reference (as html)
                   'bundle_files': 1  # bundle python25.dll in executable
                         }
              },
      zipfile=None,                   # bundle files in exe, not in library.zip
      data_files=data_files           # list of files to copy to dist directory
     )

#==============================================================================
# This section is for reference only when using older versions of matplotlib.

# The location of mpl-data files has changed across releases of matplotlib.
# Furthermore, matplotlib.get_py2exe_datafiles() had problems prior to version
# 0.99 (see link below for details), so alternative ways had to be used.
# The various techniques shown below for obtaining matplotlib auxiliary files
# (and previously used by this project) was adapted from the examples and
# discussion on http://www.py2exe.org/index.cgi/MatPlotLib.
#
# The following technique worked for matplotlib 0.91.2.
# Note that glob '*.*' will not find files that have no file extension.
'''
import glob

data_files = []
matplotlibdatadir = matplotlib.get_data_path()
mpl_lst = ('mpl-data', glob.glob(os.path.join(matplotlibdatadir, '*.*')))
data_files.append(mpl_lst)
mpl_lst = ('mpl-data', [os.path.join(matplotlibdatadir, 'matplotlibrc')])
data_files.append(mpl_lst)  # pickup file missed by glob
mpl_lst = (r'mpl-data\fonts',
           glob.glob(os.path.join(matplotlibdatadir, r'fonts\*.*')))
data_files.append(mpl_lst)
mpl_lst = (r'mpl-data\images',
           glob.glob(os.path.join(matplotlibdatadir, r'images\*.*')))
data_files.append(mpl_lst)
'''

# The following technique worked for matplotlib 0.98.5.
# Note that glob '*.*' will not find files that have no file extension.
'''
import glob

data_files = []
matplotlibdatadir = matplotlib.get_data_path()
mpl_lst = ('mpl-data', glob.glob(os.path.join(matplotlibdatadir, '*.*')))
data_files.append(mpl_lst)
mpl_lst = ('mpl-data', [os.path.join(matplotlibdatadir, 'matplotlibrc')])
data_files.append(mpl_lst)  # pickup file missed by glob
mpl_lst = (r'mpl-data\fonts\afm',
           glob.glob(os.path.join(matplotlibdatadir, r'fonts\afm\*.*')))
data_files.append(mpl_lst)
mpl_lst = (r'mpl-data\fonts\pdfcorefonts',
           glob.glob(os.path.join(matplotlibdatadir, r'fonts\pdfcorefonts\*.*')))
data_files.append(mpl_lst)
mpl_lst = (r'mpl-data\fonts\ttf',
           glob.glob(os.path.join(matplotlibdatadir, r'fonts\ttf\*.*')))
data_files.append(mpl_lst)
mpl_lst = (r'mpl-data\images',
           glob.glob(os.path.join(matplotlibdatadir, r'images\*.*')))
data_files.append(mpl_lst)
'''

# The following technique worked for matplotlib 0.98 and 0.99.
'''
from distutils.filelist import findall

data_files = []
matplotlibdatadir = matplotlib.get_data_path()
matplotlibdata = findall(matplotlibdatadir)

for f in matplotlibdata:
    dirname = os.path.join('mpl-data', f[len(matplotlibdatadir)+1:])
    data_files.append((os.path.split(dirname)[0], [f]))
'''
