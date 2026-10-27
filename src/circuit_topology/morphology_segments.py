"""将SWC形态转换为NEURON区段分段的工具。 / Utilities for converting SWC morphologies to NEURON section segments."""

import numpy as np

# =========================
# =========================
TYPE_SOMA = 1
TYPE_AXON = 2
TYPE_DEND = 3
# 已注释的顶树突类型常量定义。 / TYPE_APIC = 4


def morph2Segs(swc_path, max_seg_len=20.0, allowed_types=(TYPE_DEND,)):
    """将SWC形态转换为胞体与树突分段数组。 / Convert an SWC morphology into soma/dendrite segment arrays."""
    data = np.loadtxt(swc_path)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError("SWC input must contain at least seven columns")

    node_ids = data[:, 0].astype(int)
    node_types = data[:, 1].astype(int)
    parents = data[:, 6].astype(int)

    id2row = {int(nid): i for i, nid in enumerate(node_ids)}

    soma_ids = node_ids[node_types == TYPE_SOMA].astype(int)
    if soma_ids.size == 0:
        raise ValueError("SWC input does not contain a soma node (type 1)")

    def get_child(node_id: int):
        node_id = int(node_id)
        mask_child = (parents == node_id)
        mask_type = np.isin(node_types, np.array(allowed_types, dtype=int))
        return node_ids[mask_child & mask_type].astype(int)

    def path_length(branch_ids: np.ndarray) -> float:
        coords = []
        for nid in branch_ids:
            ridx = id2row.get(int(nid), None)
            if ridx is None:
                return np.inf
            coords.append(data[ridx, 2:5])
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))

    def advance_segment(branch: np.ndarray) -> np.ndarray:
        node = int(branch[-1])
        children = get_child(node)

        if len(children) == 1:
            nxt = int(children[0])
            if len(branch) == 1:
                branch = np.append(branch, nxt)
                return advance_segment(branch)
            else:
                L = path_length(branch)
                if L <= max_seg_len:
                    branch = np.append(branch, nxt)
                    return advance_segment(branch)

        if len(branch) > 1:
            branch = branch[:-1]
        return branch

    def get_segments(node_list, parentNode, parentSegment, segs, cons):
        seg_index = len(segs)

        # 分段由父节点与递归推进得到的节点列表组成。 / seg: [parentNode] + advance_segment([node])
        seg_path = np.append(int(parentNode), advance_segment(np.array(node_list, dtype=int)))
        segs.append(seg_path)
        cons.append([int(parentSegment), int(seg_index)])

        last_node = int(segs[-1][-1])
        child = get_child(last_node).tolist()

        if len(child) > 0:
            segs, cons = get_segments([child[0]], segs[seg_index][-1], seg_index, segs, cons)
            for j in range(1, len(child)):
                segs, cons = get_segments([child[j]], segs[seg_index][-1], seg_index, segs, cons)

        return segs, cons

    segs, cons = [], []
    for soma_id in soma_ids:
        children = get_child(int(soma_id))
        for ch in children:
            segs, cons = get_segments([int(ch)], int(soma_id), -1, segs, cons)

    soma_rows = [id2row[int(sid)] for sid in soma_ids if int(sid) in id2row]
    soma_nodes = data[soma_rows, :]
    soma_mean = soma_nodes[:, 2:5].mean(axis=0)
    avg_soma_radius = float(np.mean(soma_nodes[:, 5])) if soma_nodes.shape[0] > 0 else 1.0

    # 输出处理。 / Output handling.
    soma_seg = np.array([
        0, 0, 0, avg_soma_radius / 2,
        avg_soma_radius / np.sqrt(3), avg_soma_radius / np.sqrt(3), avg_soma_radius / np.sqrt(3), avg_soma_radius / 2,
        TYPE_SOMA, TYPE_SOMA
    ], dtype=float)

    pts_list = [soma_seg]

    # 路径与配置。 / Paths and configuration.
    for seg in segs:
        start_id = int(seg[0])
        end_id = int(seg[-1])

        if start_id not in id2row or end_id not in id2row:
            continue

        srow = id2row[start_id]
        erow = id2row[end_id]

        c0 = data[srow, 2:6]  # 三维坐标x、y、z与半径r。 / x,y,z,r
        c1 = data[erow, 2:6]
        t0 = data[srow, 1]
        t1 = data[erow, 1]

        if int(t0) == TYPE_AXON or int(t1) == TYPE_AXON:
            continue

        pts = np.concatenate([c0, c1, [t0], [t1]]).astype(float)
        pts_list.append(pts)

    pts3d = np.vstack(pts_list) if len(pts_list) > 0 else soma_seg[None, :]

    return pts3d, cons, segs, data


def axis_equal(ax):
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)
