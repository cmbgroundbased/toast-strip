# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from astropy import units as u

from .. import ops as ops

from .. import qarray as qa

from ..intervals import Interval, IntervalList

from ..observation import default_values as defaults
from ..observation import Observation

from ..data import Data

from ..pixels_io import write_wcs_fits

from ._helpers import (
    create_outdir,
    create_ground_data,
    create_comm,
    create_space_telescope,
    create_fake_sky,
)

from .mpi import MPITestCase


class PointingWCSTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_projection_default(self):
        toastcomm = create_comm(self.comm)
        data = Data(toastcomm)
        tele = create_space_telescope(
            toastcomm.group_size,
            sample_rate=1.0 * u.Hz,
            pixel_per_process=1,
        )

        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        pixels = ops.PixelsWCS(
            detector_pointing=detpointing_radec,
            create_dist="dist",
            use_astropy=True,
        )
        wcs = pixels.wcs

        # Make some fake boresight pointing
        npix_ra = pixels.pix_ra
        npix_dec = pixels.pix_dec
        px = list()
        for ra in range(npix_ra):
            px.extend(
                np.column_stack(
                    [
                        ra * np.ones(npix_dec, dtype=np.int64),
                        np.arange(npix_dec, dtype=np.int64),
                    ]
                ).tolist()
            )
        coord = wcs.wcs_pix2world(px, 0)
        bore = qa.from_position(coord[:, 1], coord[:, 0])

        nsamp = npix_ra * npix_dec
        data.obs.append(Observation(toastcomm, tele, n_samples=nsamp))
        data.obs[0].shared.create_column(
            defaults.boresight_radec, (nsamp, 4), dtype=np.float64
        )
        data.obs[0].shared.create_column(
            defaults.shared_flags, (nsamp,), dtype=np.uint8
        )
        if toastcomm.group_rank == 0:
            data.obs[0].shared[defaults.boresight_radec].set(bore)
        else:
            data.obs[0].shared[defaults.boresight_radec].set(None)

        pixels.apply(data)

        outfile = os.path.join(self.outdir, "default.fits")

    def test_mapmaking(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Compute pixels
        pixels = ops.PixelsWCS(
            detector_pointing=detpointing_radec,
            use_astropy=True,
        )

        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)
