import os
from pathlib import Path

from neuron import h, load_mechanisms
import numpy as np
import LFPy
import csv

from . import morphology_segments

from scipy.stats import poisson
from math import ceil
import json


# =========================
# =========================
TYPE_SOMA = 1
TYPE_AXON = 2
TYPE_DEND = 3


class cortical_neuron():
    """Morphology-based compartmental neuron used by the LFP simulations."""

    mechanismFolder = str(
        Path(os.environ.get(
            "MECHANISM_ROOT",
            Path(__file__).resolve().parents[2] / "mechanisms",
        )).resolve()
    )

    def __init__(self, mFile, parameters):
        pars = parameters['biophys_pars']

        AXIAL_RESISTANCE = pars['pas_mem_pars']['Ra']
        EREV_LEAK = pars['pas_mem_pars']['erev_leak']
        G_LEAK = pars['pas_mem_pars']['g_leak']

        pts3d, connections, segs, morphData = morphology_segments.morph2Segs(mFile)

        load_mechanisms(cortical_neuron.mechanismFolder)

        secType = dict()

        self.sectionList = h.SectionList()

        # Path.stem is portable across Linux and Windows path separators.
        mType = Path(mFile).stem

        # ======================================================
        # 1) Soma
        # ======================================================
        self.soma = h.Section(name='soma')
        secType['soma'] = TYPE_SOMA

        self.soma.insert('Leak')
        self.soma.e_Leak = EREV_LEAK
        self.soma.gmax_Leak = G_LEAK

        if pars.get('activeSoma', False):
            self.soma.insert('inaT')
            self.soma.shift_inaT = pars['shift'][mType]
            self.soma.vtraub_inaT = -68
            self.soma.gnabar_inaT = pars['gnabar'][mType]
            self.soma.ena = 60

            self.soma.insert('ikdT')
            self.soma.gkbar_ikdT = pars['gkbar'][mType]
            self.soma.ek = -80

        self.soma.push()
        h.pt3dclear()

        xSoma = morphData[morphData[:, 1] == TYPE_SOMA, 2:6]  # x,y,z,r
        if xSoma.size == 0:
            raise ValueError(f"[SWC TYPE ERROR] No soma points (type={TYPE_SOMA}) found in: {mFile}")

        for x in xSoma:
            h.pt3dadd(float(x[0]), float(x[1]), float(x[2]), float(x[3]))

        self.sectionList.append()
        h.pop_section()

        # ======================================================
        # ======================================================
        self.dend = []
        self.axon = []

        seg_to_sec = {}

        for i in range(len(segs)):
            node_types = []
            for j in segs[i]:
                idx = np.argwhere(morphData[:, 0] == j)[0][0]
                node_types.append(int(morphData[idx, 1]))

            if len(node_types) == 0:
                major_type = TYPE_DEND
            else:
                vals, counts = np.unique(node_types, return_counts=True)
                major_type = int(vals[np.argmax(counts)])
                if TYPE_DEND in vals and TYPE_AXON in vals:
                    c_d = counts[np.where(vals == TYPE_DEND)][0]
                    c_a = counts[np.where(vals == TYPE_AXON)][0]
                    if c_d == c_a:
                        major_type = TYPE_DEND

            if major_type == TYPE_AXON:
                secName = 'axon' + str(i)
                sec = h.Section(name=secName)
                self.axon.append(sec)
            else:
                secName = 'dend' + str(i)
                sec = h.Section(name=secName)
                self.dend.append(sec)

            seg_to_sec[i] = sec
            secType[secName] = major_type

            sec.push()
            h.pt3dclear()

            for j in segs[i]:
                idx = np.argwhere(morphData[:, 0] == j)[0][0]
                h.pt3dadd(float(morphData[idx, 2]),
                          float(morphData[idx, 3]),
                          float(morphData[idx, 4]),
                          float(morphData[idx, 5]))

            sec.insert('Leak')
            sec.e_Leak = EREV_LEAK
            sec.gmax_Leak = G_LEAK
            sec.Ra = AXIAL_RESISTANCE

            if major_type == TYPE_DEND:
                self.sectionList.append()

            h.pop_section()

        # ======================================================
        # ======================================================
        for i in range(len(connections)):
            i1 = int(connections[i][0])  # parent seg index
            i0 = int(connections[i][1])  # child seg index

            child_sec = seg_to_sec.get(i0, None)
            if child_sec is None:
                continue

            if i1 == -1:
                child_sec.connect(self.soma, 1, 0)
            else:
                parent_sec = seg_to_sec.get(i1, None)
                if parent_sec is None:
                    child_sec.connect(self.soma, 1, 0)
                else:
                    child_sec.connect(parent_sec(1))
