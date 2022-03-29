# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from astropy import units as u

import warnings

from pixell import wcsutils

from ..utils import Environment, Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance, Tuple

from ..timing import function_timer

from .. import qarray as qa

from ..pixels import PixelDistribution

from .operator import Operator

from .delete import Delete


@trait_docs
class PixelsWCS(Operator):
    """Operator which generates detector pixel indices defined on a flat projection.

    When placing the projection on the sky, either the `wcs_center` or `wcs_bounds`
    traits must be specified, but not both.

    When determining the pixel density in the projection, two traits from the set of
    `wcs_bounds`, `wcs_resolution` and `wcs_dimensions` must be specified.

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels.

    This uses the pixell package to construct the WCS projection parameters.  By
    default, the world to pixel conversion is performed with internal, optimized code
    unless use_astropy is set to True.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    wcs_projection = Unicode(
        "CAR", help="Supported values are CAR, CEA, MER, ZEA, AIR, TAN"
    )

    wcs_center = Tuple(
        (0 * u.degree, 0 * u.degree),
        allow_none=True,
        help="The center coordinates (Quantities) of the projection",
    )

    wcs_bounds = Tuple(
        None,
        allow_none=True,
        help="The lower left and upper right corners (Quantities)",
    )

    wcs_dimensions = Tuple(
        (360, 180),
        allow_none=True,
        help="The RA/DEC pixel dimensions of the projection",
    )

    wcs_resolution = Tuple(
        (1 * u.degree, 1 * u.degree),
        allow_none=True,
        help="Projection resolution (Quantities) along the 2 axes",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode("pixels", help="Observation detdata key for output pixel indices")

    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output quaternions",
    )

    submaps = Int(256, help="Number of submaps to use")

    create_dist = Unicode(
        None,
        allow_none=True,
        help="Create the submap distribution for all detectors and store in the Data key specified",
    )

    single_precision = Bool(False, help="If True, use 32bit int in output")

    use_astropy = Bool(False, help="If True, use astropy for world to pix conversion")

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("wcs_projection")
    def _check_wcs_projection(self, proposal):
        check = proposal["value"]
        if check not in ["CAR", "CEA", "MER", "ZEA", "AIR", "TAN"]:
            raise traitlets.TraitError("Invalid WCS projection name")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # If running with all default values, the 'observe' function will not
        # have been called yet.
        if not hasattr(self, "_local_submaps"):
            self._set_wcs(
                self.wcs_projection,
                self.wcs_center,
                self.wcs_resolution,
                self.wcs_dimensions,
            )

    @traitlets.observe(
        "wcs_projection", "wcs_center", "wcs_bounds", "wcs_dimensions", "wcs_resolution"
    )
    def _reset_wcs(self, change):
        # (Re-)initialize the WCS projection when one of these traits change.
        # Current values:
        proj = self.wcs_projection
        center = self.wcs_center
        bounds = self.wcs_bounds
        dims = self.wcs_dimensions
        res = self.wcs_resolution
        pos = None
        if center is not None:
            pos = center
        else:
            pos = bounds

        # Update to the trait that changed
        if change["name"] == "wcs_projection":
            proj = change["new"]
        if change["name"] == "wcs_center":
            center = change["new"]
            bounds = None
            pos = center
        if change["name"] == "wcs_bounds":
            bounds = change["new"]
            center = None
            pos = bounds
        if change["name"] == "wcs_dimensions":
            dims = change["new"]
        if change["name"] == "wcs_resolution":
            res = change["new"]
        self._set_wcs(proj, pos, res, dims)
        self.wcs_projection = proj
        self.wcs_center = center
        self.wcs_bounds = bounds
        self.wcs_dimensions = dims
        self.wcs_resolution = res

    def _set_wcs(self, proj, pos, res, shape):
        res = np.array(
            [
                res[0].to_value(u.degree),
                res[1].to_value(u.degree),
            ]
        )
        if isinstance(pos[0], u.Quantity):
            # Center
            pos = np.array(
                [
                    pos[0].to_value(u.degree),
                    pos[1].to_value(u.degree),
                ]
            )
        else:
            # Bounds
            pos = np.array(
                [
                    [pos[0][0].to_value(u.degree), pos[0][1].to_value(u.degree)],
                    [pos[1][0].to_value(u.degree), pos[1][1].to_value(u.degree)],
                ]
            )
        if proj == "CAR":
            self.wcs = wcsutils.car(pos, res=res, shape=shape)
        elif proj == "CEA":
            self.wcs = wcsutils.cea(pos, res=res, shape=shape)
        elif proj == "MER":
            self.wcs = wcsutils.mer(pos, res=res, shape=shape)
        elif proj == "ZEA":
            self.wcs = wcsutils.zea(pos, res=res, shape=shape)
        elif proj == "TAN":
            self.wcs = wcsutils.tan(pos, res=res, shape=shape)
        elif proj == "AIR":
            self.wcs = wcsutils.air(pos, res=res, shape=shape)
        else:
            raise RuntimeError("Unsupported projection")
        self.pix_ra = int(np.round(2 * self.wcs.wcs.crpix[0] - 1))
        self.pix_dec = int(np.round(2 * self.wcs.wcs.crpix[1] - 1))
        self._n_pix = self.pix_ra * self.pix_dec
        self._n_pix_submap = self._n_pix // self.submaps
        if self._n_pix_submap * self.submaps < self._n_pix:
            self._n_pix_submap += 1
        self._local_submaps = None

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        if self._local_submaps is None and self.create_dist is not None:
            self._local_submaps = np.zeros(self.submaps, dtype=np.bool)

        if not self.use_astropy:
            raise NotImplementedError("Only astropy conversion is currently supported")

        # Expand detector pointing
        if self.quats is not None:
            quats_name = self.quats
        else:
            if self.detector_pointing.quats is not None:
                quats_name = self.detector_pointing.quats
            else:
                quats_name = "quats"

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        self.detector_pointing.quats = quats_name
        self.detector_pointing.apply(data, detectors=detectors)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Create (or re-use) output data for the pixels, weights and optionally the
            # detector quaternions.

            if self.single_precision:
                exists = ob.detdata.ensure(
                    self.pixels, sample_shape=(), dtype=np.int32, detectors=dets
                )
            else:
                exists = ob.detdata.ensure(
                    self.pixels, sample_shape=(), dtype=np.int64, detectors=dets
                )

            # Do we already have pointing for all requested detectors?
            if exists:
                # Yes...
                if self.create_dist is not None:
                    # but the caller wants the pixel distribution
                    for ob in data.obs:
                        views = ob.view[self.view]
                        for det in ob.select_local_detectors(detectors):
                            for view in range(len(views)):
                                self._local_submaps[
                                    views.detdata[self.pixels][view][det]
                                    // self._n_pix_submap
                                ] = True

                if data.comm.group_rank == 0:
                    msg = (
                        f"Group {data.comm.group}, ob {ob.name}, WCS pixels "
                        f"already computed for {dets}"
                    )
                    log.verbose(msg)
                continue

            # Focalplane for this observation
            focalplane = ob.telescope.focalplane

            # Loop over views
            views = ob.view[view]
            for vw in range(len(views)):
                # Get the flags if needed.  Use the same flags as
                # detector pointing.
                flags = None
                if self.detector_pointing.shared_flags is not None:
                    flags = np.array(
                        views.shared[self.detector_pointing.shared_flags][vw]
                    )
                    flags &= self.detector_pointing.shared_flag_mask

                for det in dets:
                    # Timestream of detector quaternions
                    quats = views.detdata[quats_name][vw][det]
                    view_samples = len(quats)

                    theta, phi = qa.to_position(quats)
                    rdpix = self.wcs.wcs_world2pix(
                        np.column_stack([np.rad2deg(phi), 90 - np.rad2deg(theta)]), 0
                    )
                    views.detdata[self.pixels][vw][det] = (
                        rdpix[:, 0] * self.pix_dec + rdpix[:, 1]
                    )

                    if self.create_dist is not None:
                        self._local_submaps[
                            views.detdata[self.pixels][vw][det] // self._n_pix_submap
                        ] = 1

    def _finalize(self, data, **kwargs):
        if self.create_dist is not None:
            submaps = None
            if self.single_precision:
                submaps = np.arange(self.submaps, dtype=np.int32)[
                    self._local_submaps == 1
                ]
            else:
                submaps = np.arange(self.submaps, dtype=np.int64)[
                    self._local_submaps == 1
                ]

            data[self.create_dist] = PixelDistribution(
                n_pix=self._n_pix,
                n_submap=self.submaps,
                local_submaps=submaps,
                comm=data.comm.comm_world,
            )
            # Store a copy of the WCS information in the distribution object
            data[self.create_dist].wcs = self.wcs.deepcopy()
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prov["detdata"].append(self.pixels)
        if self.create_dist is not None:
            prov["global"].append(self.create_dist)
        return prov
