#    Copyright (C) 2004-2011 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time

basedir = os.path.abspath(os.path.split(__file__)[0])


def get_revision():
    """Returns revision and vcs information, dynamically obtained."""
    vcs, revision, tag = None, None, None

    hgdir = os.path.join(basedir, "..", ".hg")
    gitdir = os.path.join(basedir, "..", ".git")

    if os.path.isdir(hgdir):
        vcs = "mercurial"
        try:
            p = subprocess.Popen(["hg", "id"], cwd=basedir, stdout=subprocess.PIPE)
        except OSError:
            # Could not run hg, even though this is a mercurial repository.
            pass
        else:
            stdout = p.communicate()[0]
            # Force strings instead of unicode.
            x = list(map(str, stdout.decode().strip().split()))

            if len(x) == 0:
                # Somehow stdout was empty. This can happen, for example,
                # if you're running in a terminal which has redirected stdout.
                # In this case, we do not use any revision/tag info.
                pass
            elif len(x) == 1:
                # We don't have 'tip' or anything similar...so no tag.
                revision = str(x[0])
            else:
                revision = str(x[0])
                tag = str(x[1])

    elif os.path.isdir(gitdir):
        vcs = "git"
        # For now, we are not bothering with revision and tag.

    vcs_info = (vcs, (revision, tag))

    return revision, vcs_info


def get_info(dynamic=True):
    # Date information
    date_info = datetime.datetime.now()
    date = time.asctime(date_info.timetuple())

    revision, version, version_info, vcs_info = None, None, None, None

    import_failed = False
    dynamic_failed = False

    if dynamic:
        revision, vcs_info = get_revision()
        if revision is None:
            dynamic_failed = True

    if dynamic_failed or not dynamic:
        # This is where most final releases of NetworkX will be.
        # All info should come from version.py. If it does not exist, then
        # no vcs information will be provided.
        sys.path.insert(0, basedir)
        try:
            from version import date, date_info, vcs_info, version, version_info  # type: ignore
        except ImportError:
            import_failed = True
            vcs_info = (None, (None, None))
        else:
            revision = vcs_info[1][0]
        del sys.path[0]

    if import_failed or (dynamic and not dynamic_failed):
        # We are here if:
        #   we failed to determine static versioning info, or
        #   we successfully obtained dynamic revision info
        version = "".join([str(major), ".", str(minor)])  # noqa
        if dev:
            version += ".dev_" + date_info.strftime("%Y%m%d%H%M%S")
        version_info = (name, major, minor, revision)  # noqa

    return date, date_info, version, version_info, vcs_info


# Version information
name = "networkx"
major = "1"
minor = "8.1"


# Declare current release as a development release.
# Change to False before tagging a release; then change back.
dev = False


description = "Python package for creating and manipulating graphs and networks"

long_description = """
NetworkX is a Python package for the creation, manipulation, and
study of the structure, dynamics, and functions of complex networks.

"""
license = "BSD"
authors = {
    "Hagberg": ("Aric Hagberg", "hagberg@lanl.gov"),
    "Schult": ("Dan Schult", "dschult@colgate.edu"),
    "Swart": ("Pieter Swart", "swart@lanl.gov"),
}
maintainer = "NetworkX Developers"
maintainer_email = "networkx-discuss@googlegroups.com"
url = "http://networkx.lanl.gov/"
download_url = "http://networkx.lanl.gov/download/networkx"
platforms = ["Linux", "Mac OSX", "Windows", "Unix"]
keywords = [
    "Networks",
    "Graph Theory",
    "Mathematics",
    "network",
    "graph",
    "discrete mathematics",
    "math",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.6",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.1",
    "Programming Language :: Python :: 3.2",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
]

date, date_info, version, version_info, vcs_info = get_info()
