# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np

from astropy import units as u

from astropy.table import Table, Column

from .timing import function_timer, Timer

from .utils import Logger, Environment


class Scan(object):
    """Base class for simulated telescope scan properties for one observation.

    We use python datetime for specifying times.  These are trivially convertable to
    astrometry packages.

    Args:
        name (str):  Arbitrary name (does not have to be unique).
        start (datetime):  The start time of the scan.
        stop (datetime):  The stop time of the scan.
    """

    def __init__(self, name=None, start=None, stop=None):
        self.name = name
        if start is None:
            raise RuntimeError("you must specify the start time")
        if stop is None:
            raise RuntimeError("you must specify the stop time")
        self.start = start
        self.stop = stop


class GroundScan(Scan):
    """Simulated ground telescope scan properties for one observation.

    Args:
        name (str):  Arbitrary name (does not have to be unique).
        start (datetime):  The start time of the scan.
        stop (datetime):  The stop time of the scan.
        boresight_angle (Quantity):  Boresight rotation angle.
        az_min (Quantity):  The minimum Azimuth value of each sweep.
        az_max (Quantity):  The maximum Azimuth value of each sweep.
        el (Quantity):  The nominal Elevation of the scan.
        rising (bool):  If True, the patch is rising, else it is setting.
        sun_az_begin (Quantity):  The Sun Azimuth value at the beginning of the scan.
        sun_az_end (Quantity):  The Sun Azimuth value at the end of the scan.
        sun_el_begin (Quantity):  The Sun Elevation value at the beginning of the scan.
        sun_el_end (Quantity):  The Sun Elevation value at the end of the scan.
        moon_az_begin (Quantity):  The Moon Azimuth value at the beginning of the scan.
        moon_az_end (Quantity):  The Moon Azimuth value at the end of the scan.
        moon_el_begin (Quantity):  The Moon Elevation value at the beginning of the
            scan.
        moon_el_end (Quantity):  The Moon Elevation value at the end of the scan.
        moon_phase (float):  The phase of the moon as a value from 0 to 1.
        scan_indx (int):  The current pass of this patch in the overall schedule.
        subscan_indx (int):  The current sub-pass of this patch in the overall schedule.

    """

    def __init__(
        self,
        name=None,
        start=None,
        stop=None,
        boresight_angle=0 * u.degree,
        az_min=0 * u.degree,
        az_max=0 * u.degree,
        el=0 * u.degree,
        rising=False,
        sun_az_begin=0 * u.degree,
        sun_az_end=0 * u.degree,
        sun_el_begin=0 * u.degree,
        sun_el_end=0 * u.degree,
        moon_az_begin=0 * u.degree,
        moon_az_end=0 * u.degree,
        moon_el_begin=0 * u.degree,
        moon_el_end=0 * u.degree,
        moon_phase=0.0,
        scan_indx=0,
        subscan_indx=0,
    ):
        super().__init__(name=name, start=start, stop=stop)
        self.boresight_angle = boresight_angle
        self.az_min = az_min
        self.az_max = az_max
        self.el = el
        self.rising = rising
        self.sun_az_begin = sun_az_begin
        self.sun_az_end = sun_az_end
        self.sun_el_begin = sun_el_begin
        self.sun_el_end = sun_el_end
        self.moon_az_begin = moon_az_begin
        self.moon_az_end = moon_az_end
        self.moon_el_begin = moon_el_begin
        self.moon_el_end = moon_el_end
        self.moon_phase = moon_phase
        self.scan_indx = scan_indx
        self.subscan_indx = subscan_indx

    def min_sso_dist(self, sso_az_begin, sso_el_begin, sso_az_end, sso_el_end):
        """Rough minimum angle between the boresight and a solar system object.

        Args:
            sso_az_begin (Quantity):  Object starting Azimuth
            sso_el_begin (Quantity):  Object starting Elevation
            sso_az_end (Quantity):  Object final Azimuth
            sso_el_end (Quantity):  Object final Elevation

        Returns:
            (Quantity):  The minimum angle.

        """
        sso_vec1 = hp.dir2vec(
            sso_az_begin.to_value(u.degree),
            sso_el_begin.to_value(u.degree),
            lonlat=True,
        )
        sso_vec2 = hp.dir2vec(
            sso_az_end.to_value(u.degree), sso_el_end.to_value(u.degree), lonlat=True
        )
        az1 = self.az_min.to_value(u.degree)
        az2 = self.az_max.to_value(u.degree)
        if az2 < az1:
            az2 += 360.0
        n = 100
        az = np.linspace(az1, az2, n)
        el = np.ones(n) * self.el.to_value(u.degree)
        vec = hp.dir2vec(az, el, lonlat=True)
        dist1 = np.degrees(np.arccos(np.dot(sso_vec1, vec)))
        dist2 = np.degrees(np.arccos(np.dot(sso_vec2, vec)))
        result = min(np.amin(dist1), np.amin(dist2))
        return result * u.degree


class SatelliteScan(Scan):
    """Simulated satellite telescope scan properties for one observation.

    This class assumes a simplistic model where the nominal precession axis is pointing
    in the anti-sun direction (from a location such as at L2).  This class just
    specifies the rotation rates about this axis and also about the spin axis.  The
    opening angles are part of the Telescope and not specified here.

    Args:
        name (str):  Arbitrary name (does not have to be unique).
        start (datetime):  The start time of the scan.
        stop (datetime):  The stop time of the scan.
        prec_period (Quantity):  The time for one revolution about the precession axis.
        spin_period (Quantity):  The time for one revolution about the spin axis.

    """

    def __init__(
        self,
        name=None,
        start=None,
        stop=None,
        prec_period=0 * u.minute,
        spin_period=0 * u.minute,
    ):
        super().__init__(name=name, start=start, stop=stop)
        self.prec_period = prec_period
        self.spin_period = spin_period


class GroundSchedule(object):
    """Class representing a ground based observing schedule.

    A schedule is a collection of scans, with some extra methods for doing I/O.

    Args:
        scans (list):  A list of Scan instances or None.

    """

    def __init__(self, scans=None):
        self.scans = scans
        if scans is None:
            self.scans = list()

    @function_timer
    def read(self, file, file_split=False, sort=False):
        """Load a ground observing schedule from a file.

        This loads scans from a file and appends them to the internal list of scans.
        The resulting combined scan list is optionally sorted.

        Args:
            file (str):  The file to load.
            file_split (tuple):  If not None, only use a subset of the schedule file.
                The arguments are (isplit, nsplit) and only observations that satisfy
                'scan index modulo nsplit == isplit' are included.
            sort (bool):  If True, sort the combined scan list by name.

        Returns:
            None

        """

        def _parse_line(line):
            """Parse one line of the schedule file"""
            if line.startswith("#"):
                return None
            fields = line.split()
            nfield = len(fields)
            if nfield == 22:
                # Deprecated prior to 2020-02 schedule format without boresight rotation field
                (
                    start_date,
                    start_time,
                    stop_date,
                    stop_time,
                    mjdstart,
                    mjdstop,
                    name,
                    azmin,
                    azmax,
                    el,
                    rs,
                    sun_el1,
                    sun_az1,
                    sun_el2,
                    sun_az2,
                    moon_el1,
                    moon_az1,
                    moon_el2,
                    moon_az2,
                    moon_phase,
                    scan,
                    subscan,
                ) = line.split()
                boresight_angle = 0
            else:
                # 2020-02 schedule format with boresight rotation field
                (
                    start_date,
                    start_time,
                    stop_date,
                    stop_time,
                    mjdstart,
                    mjdstop,
                    boresight_angle,
                    name,
                    azmin,
                    azmax,
                    el,
                    rs,
                    sun_el1,
                    sun_az1,
                    sun_el2,
                    sun_az2,
                    moon_el1,
                    moon_az1,
                    moon_el2,
                    moon_az2,
                    moon_phase,
                    scan,
                    subscan,
                ) = line.split()
            start_time = start_date + " " + start_time
            stop_time = stop_date + " " + stop_time
            try:
                start_time = dateutil.parser.parse(start_time + " +0000")
                stop_time = dateutil.parser.parse(stop_time + " +0000")
            except Exception:
                start_time = dateutil.parser.parse(start_time)
                stop_time = dateutil.parser.parse(stop_time)
            return GroundScan(
                name,
                start_time,
                stop_time,
                float(boresight_angle) * u.degree,
                float(azmin) * u.degree,
                float(azmax) * u.degree,
                float(el) * u.degree,
                (rs.upper() == "R"),
                float(sun_az1) * u.degree,
                float(sun_az2) * u.degree,
                float(sun_el1) * u.degree,
                float(sun_el2) * u.degree,
                float(moon_az1) * u.degree,
                float(moon_az2) * u.degree,
                float(moon_el1) * u.degree,
                float(moon_el2) * u.degree,
                float(moon_phase),
                scan,
                subscan,
            )

        isplit = None
        nsplit = None
        if file_split is not None:
            isplit, nsplit = file_split
        scan_counters = dict()

        with open(file, "r") as f:
            while True:
                line = f.readline()
                if line.startswith("#"):
                    continue
                (
                    site_name,
                    telescope_name,
                    site_lat,
                    site_lon,
                    site_alt,
                ) = line.split()
                break
            last_name = None
            for line in f:
                if line.startswith("#"):
                    continue
                gscan = _parse_line(line)
                if nsplit is not None:
                    # Only accept 1 / `nsplit` of the rising and setting
                    # scans in patch `name`.  Selection is performed
                    # during the first subscan.
                    if name != last_name:
                        if name not in scan_counters:
                            scan_counters[name] = dict()
                        counter = scan_counters[name]
                        # Separate counters for rising and setting scans
                        ckey = "S"
                        if gscan.rising:
                            ckey = "R"
                        if ckey not in counter:
                            counter[ckey] = 0
                        else:
                            counter[ckey] += 1
                        iscan = counter[ckey]
                    last_name = name
                    if iscan % nsplit != isplit:
                        continue
                self.scans.append(gscan)
        if sort:
            sortedscans = sorted(self.scans, key=lambda scn: scn.name)
            self.scans = sortedscans

    @function_timer
    def write(self, file):
        # FIXME:  We should have more robust format here (e.g. ECSV) and then use
        # This class when building the schedule as well.
        raise NotImplementedError("New ground schedule format not yet implemented")


class SatelliteSchedule(object):
    """Class representing a satellite observing schedule.

    A schedule is a collection of scans, with some extra methods for doing I/O.

    Args:
        scans (list):  A list of Scan instances or None.

    """

    def __init__(self, scans=None):
        self.scans = scans
        if scans is None:
            self.scans = list()

    @function_timer
    def read(self, file, sort=False):
        """Load a satellite observing schedule from a file.

        This loads scans from a file and appends them to the internal list of scans.
        The resulting combined scan list is optionally sorted.

        Args:
            file (str):  The file to load.
            sort (bool):  If True, sort the combined scan list by name.

        Returns:
            None

        """
        pass

    @function_timer
    def write(self, file):
        """Write satellite schedule to a file.

        This writes the internal scan list to the specified file.

        Args:
            file (str):  The file to write.

        Returns:
            None

        """
        n_rows = len(self.scans)

        # The max number of characters to represent an ISO 8601 format time
        # with a timezone and microsecond resolution.
        max_tstr = 32

        cols = [
            Column(
                name="name",
                length=nrows,
                dtype=np.dtype("a20"),
                description="The name of the scan",
            ),
            Column(
                name="name",
                length=nrows,
                dtype=np.dtype("a20"),
                description="The timestamp of the event (UTC, ISO format)",
            ),
            Column(
                name="PETAL",
                length=nrows,
                dtype=np.int32,
                description="Petal location [0-9]",
            ),
            Column(
                name="DEVICE",
                length=nrows,
                dtype=np.int32,
                description="Device location on the petal",
            ),
            Column(
                name="LOCATION",
                length=nrows,
                dtype=np.int32,
                description="Global device location (PETAL * 1000 + DEVICE)",
            ),
            Column(
                name="STATE",
                length=nrows,
                dtype=np.uint32,
                description="State bit field (good == 0)",
            ),
            Column(
                name="EXCLUSION",
                length=nrows,
                dtype=np.dtype("a9"),
                description="The exclusion polygon for this device",
            ),
        ]

    out_state = Table()
    out_state.add_columns(out_cols)

     out_state_file = os.path.join(
        outdir, "desi-state_{}.ecsv".format(file_date))

    out_fp.write(out_fp_file, format="ascii.ecsv", overwrite=True)
