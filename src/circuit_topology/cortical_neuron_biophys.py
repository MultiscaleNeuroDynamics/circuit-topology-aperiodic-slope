# -*- coding: utf-8 -*-
"""Passive multicompartment neuron with explicit Ra, leak and cm controls."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from neuron import h, load_mechanisms

from . import morphology_segments

TYPE_SOMA, TYPE_AXON, TYPE_DEND = 1, 2, 3


class cortical_neuron:
    mechanismFolder = str(
        Path(os.environ.get(
            "MECHANISM_ROOT",
            Path(__file__).resolve().parents[2] / "mechanisms",
        )).resolve()
    )

    def __init__(self, mFile, parameters):
        pars = parameters["biophys_pars"]
        pas = pars["pas_mem_pars"]
        ra = float(pas["Ra"])
        erev = float(pas["erev_leak"])
        gleak = float(pas["g_leak"])
        cm = float(pas.get("cm", 1.0))
        _, connections, segs, morph = morphology_segments.morph2Segs(mFile)
        load_mechanisms(self.mechanismFolder)
        self.sectionList = h.SectionList()
        self.soma = h.Section(name="soma")
        self.soma.insert("Leak")
        self.soma.e_Leak, self.soma.gmax_Leak = erev, gleak
        self.soma.Ra, self.soma.cm = ra, cm
        # Preserve the baseline implementation if active soma was enabled in
        # parameter.json.  The sensitivity control does not add active dendrites.
        mtype = Path(mFile).stem
        if pars.get("activeSoma", False):
            self.soma.insert("inaT")
            self.soma.shift_inaT = pars["shift"][mtype]
            self.soma.vtraub_inaT = -68
            self.soma.gnabar_inaT = pars["gnabar"][mtype]
            self.soma.ena = 60
            self.soma.insert("ikdT")
            self.soma.gkbar_ikdT = pars["gkbar"][mtype]
            self.soma.ek = -80
        self.soma.push(); h.pt3dclear()
        soma_pts = morph[morph[:, 1] == TYPE_SOMA, 2:6]
        if soma_pts.size == 0:
            raise ValueError(f"No soma points in {mFile}")
        for point in soma_pts:
            h.pt3dadd(*map(float, point))
        self.sectionList.append(); h.pop_section()

        self.dend, self.axon, seg_to_sec = [], [], {}
        for seg_index, node_ids in enumerate(segs):
            node_types = []
            for node_id in node_ids:
                row = np.argwhere(morph[:, 0] == node_id)[0][0]
                node_types.append(int(morph[row, 1]))
            if node_types:
                vals, counts = np.unique(node_types, return_counts=True)
                major = int(vals[np.argmax(counts)])
                if TYPE_DEND in vals and TYPE_AXON in vals:
                    nd = counts[np.where(vals == TYPE_DEND)][0]
                    na = counts[np.where(vals == TYPE_AXON)][0]
                    if nd == na:
                        major = TYPE_DEND
            else:
                major = TYPE_DEND
            sec = h.Section(name=("axon" if major == TYPE_AXON else "dend") + str(seg_index))
            (self.axon if major == TYPE_AXON else self.dend).append(sec)
            seg_to_sec[seg_index] = sec
            sec.push(); h.pt3dclear()
            for node_id in node_ids:
                row = np.argwhere(morph[:, 0] == node_id)[0][0]
                h.pt3dadd(*map(float, morph[row, 2:6]))
            sec.insert("Leak")
            sec.e_Leak, sec.gmax_Leak = erev, gleak
            sec.Ra, sec.cm = ra, cm
            if major == TYPE_DEND:
                self.sectionList.append()
            h.pop_section()
        for parent_index, child_index in connections:
            parent_index, child_index = int(parent_index), int(child_index)
            child = seg_to_sec.get(child_index)
            if child is None:
                continue
            parent = seg_to_sec.get(parent_index) if parent_index != -1 else None
            child.connect(parent if parent is not None else self.soma, 1, 0)
