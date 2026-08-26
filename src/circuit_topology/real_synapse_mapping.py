# -*- coding: utf-8 -*-
"""Map MICrONS patch synapses to morphology segments for each cortical layer."""

import os
import re
import numpy as np
import pandas as pd
from standard_transform import minnie_ds


def build_clean_synapse_segment_maps(
    patch_dir: str,
    swc_paths,
    out_dir: str,
    swc_layer_names=("L2/3", "L4", "L5", "L6"),
):
    """Create four two-column synapse-to-segment mapping tables."""

    # ======================================================
    # ======================================================
    layer_to_swc = {}
    if isinstance(swc_paths, dict):
        # File-handling details.
        for k, v in swc_paths.items():
            kk = str(k).strip()
            if kk in swc_layer_names:
                layer_to_swc[kk] = v
            else:
                name = os.path.basename(kk)
                if name == "23.swc":
                    layer_to_swc["L2/3"] = v
                elif name == "4.swc":
                    layer_to_swc["L4"] = v
                elif name == "5.swc":
                    layer_to_swc["L5"] = v
                elif name == "6.swc":
                    layer_to_swc["L6"] = v
    elif isinstance(swc_paths, (list, tuple)) and len(swc_paths) == 4:
        layer_to_swc = {
            "L2/3": swc_paths[0],
            "L4":   swc_paths[1],
            "L5":   swc_paths[2],
            "L6":   swc_paths[3],
        }
    else:
        raise ValueError("swc_paths must be a layer-to-path mapping or a four-item sequence.")

    os.makedirs(out_dir, exist_ok=True)

    # ======================================================
    # ======================================================
    cells_path  = os.path.join(patch_dir, "cells_patch.csv")
    syn_ee_path = os.path.join(patch_dir, "syn_ee.csv")
    syn_ie_path = os.path.join(patch_dir, "syn_ie.csv")

    for p in [cells_path, syn_ee_path, syn_ie_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required MICrONS patch input is missing: {p}")

    cells = pd.read_csv(cells_path)
    syn_ee = pd.read_csv(syn_ee_path)
    syn_ie = pd.read_csv(syn_ie_path)

    need_cells_cols = ["pt_root_id", "pt_xt", "pt_yt", "pt_zt",
                       "classification_system_auto", "cell_type_auto"]
    for c in need_cells_cols:
        if c not in cells.columns:
            raise ValueError(f"cells_patch.csv is missing required column: {c}")

    # ======================================================
    # 2) layer_auto
    # ======================================================
    def map_layer(ct):
        if pd.isna(ct):
            return "unknown"
        s = str(ct).strip()
        if re.fullmatch(r"23P", s): return "L2/3"
        if re.fullmatch(r"4P", s):  return "L4"
        if re.fullmatch(r"5P-(IT|ET|NP)", s): return "L5"
        if re.fullmatch(r"6P-(IT|CT)", s):    return "L6"
        if re.fullmatch(r"(BC|BPC|MC|NGC)", s): return "INH_misc"
        if re.fullmatch(r"(astrocyte|oligo|OPC|microglia|pericyte)", s): return "glia"
        return "other"

    if ("layer_auto" not in cells.columns) or (cells["layer_auto"].isna().all()):
        cells["layer_auto"] = cells["cell_type_auto"].apply(map_layer)

    cells_core = cells[cells["classification_system_auto"].isin(["excitatory_neuron", "inhibitory_neuron"])].copy()
    core_ids = set(cells_core["pt_root_id"].to_numpy())

    exc_core = cells_core[cells_core["classification_system_auto"] == "excitatory_neuron"]
    layers_of_interest = list(swc_layer_names)

    layer_centers = {}
    for layer in layers_of_interest:
        sub = exc_core[exc_core["layer_auto"] == layer]
        if len(sub) == 0:
            print(f"[WARN] No excitatory patch cells found for {layer}; skipping layer center.")
            continue
        layer_centers[layer] = {
            "y": float(sub["pt_yt"].median()),
            "z": float(sub["pt_zt"].median()),
            "x": float(sub["pt_xt"].median()),
        }

    cells_lookup = cells[["pt_root_id", "pt_xt", "pt_yt", "pt_zt",
                          "classification_system_auto", "layer_auto"]].copy()

    # ======================================================
    # 3) syn pre/post & synapse_id
    # ======================================================
    def pick_pre_post_cols(df):
        if "pre_pt_root_id" in df.columns and "post_pt_root_id" in df.columns:
            return "pre_pt_root_id", "post_pt_root_id"
        if "pre_root_id" in df.columns and "post_root_id" in df.columns:
            return "pre_root_id", "post_root_id"
        raise ValueError("Synapse table does not contain recognizable pre/post root-ID columns.")

    def pick_syn_id_col(df):
        for c in ["synapse_id", "pt_synapse_id", "syn_id", "id"]:
            if c in df.columns:
                return c
        return None

    pre_col_ee, post_col_ee = pick_pre_post_cols(syn_ee)
    pre_col_ie, post_col_ie = pick_pre_post_cols(syn_ie)

    syn_id_col_ee = pick_syn_id_col(syn_ee)
    syn_id_col_ie = pick_syn_id_col(syn_ie)

    # ======================================================
    # 4) post_pt_position -> post_xt/yt/zt (transform_vx)
    # ======================================================
    def parse_vec3(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) != 3:
                return (np.nan, np.nan, np.nan)
            try:
                return float(val[0]), float(val[1]), float(val[2])
            except Exception:
                return (np.nan, np.nan, np.nan)
        if isinstance(val, str):
            s = val.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            for ch in [",", ";"]:
                s = s.replace(ch, " ")
            parts = s.split()
            if len(parts) != 3:
                return (np.nan, np.nan, np.nan)
            try:
                return tuple(map(float, parts))
            except Exception:
                return (np.nan, np.nan, np.nan)
        return (np.nan, np.nan, np.nan)

    def ensure_post_xyz_transformed(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "post_xt" in d.columns and "post_yt" in d.columns and "post_zt" in d.columns:
            return d

        if "post_pt_position" in d.columns:
            xyz_arr = np.vstack(d["post_pt_position"].apply(parse_vec3).to_numpy())
            d["post_pt_position_x"] = xyz_arr[:, 0]
            d["post_pt_position_y"] = xyz_arr[:, 1]
            d["post_pt_position_z"] = xyz_arr[:, 2]

        need_cols = ["post_pt_position_x", "post_pt_position_y", "post_pt_position_z"]
        if all(c in d.columns for c in need_cols):
            pts = d[need_cols].to_numpy(dtype=float)
            good = np.isfinite(pts).all(axis=1)
            if np.any(good):
                out = np.full_like(pts, np.nan, dtype=float)
                out_good = minnie_ds.transform_vx.apply(pts[good], as_int=False)
                out[good, :] = np.asarray(out_good, dtype=float)
                d["post_xt"] = out[:, 0]
                d["post_yt"] = out[:, 1]
                d["post_zt"] = out[:, 2]
        return d

    # ======================================================
    # 5) SWC -> segments (z0,y0,z1,y1) relative soma
    # ======================================================
    TYPE_SOMA = 1
    TYPE_DEND = 3

    def id_to_row_index_map(data):
        ids = data[:, 0].astype(int)
        return {int(i): idx for idx, i in enumerate(ids)}

    def get_child(node, data, allowed_types=(TYPE_DEND,)):
        ids = data[:, 0].astype(int)
        parents = data[:, 6].astype(int)
        types = data[:, 1].astype(int)
        mask_child = (parents == int(node))
        mask_type = np.isin(types, np.array(allowed_types, dtype=int))
        return ids[mask_child & mask_type].astype(int)

    def id_to_xyz(data, node_ids):
        mp = id_to_row_index_map(data)
        coords = []
        for nid in node_ids:
            idx = mp.get(int(nid))
            if idx is None:
                raise KeyError(f"SWC node id={nid} not found")
            coords.append(data[idx, 2:5])
        return np.asarray(coords, dtype=float)

    def advance_segment(branch, data, max_length=20.0, allowed_types=(TYPE_DEND,)):
        node = int(branch[-1])
        child_ids = get_child(node, data, allowed_types=allowed_types)

        if len(child_ids) == 1:
            nxt = int(child_ids[0])
            if len(branch) == 1:
                branch = np.append(branch, nxt)
                return advance_segment(branch, data, max_length=max_length, allowed_types=allowed_types)
            coords = id_to_xyz(data, branch)
            L = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
            if L <= max_length:
                branch = np.append(branch, nxt)
                return advance_segment(branch, data, max_length=max_length, allowed_types=allowed_types)

        elif len(branch) > 1:
            branch = branch[:-1]

        return branch

    def get_segments(node, parentNode, parentSegment, segs, cons, data,
                     max_length=20.0, allowed_types=(TYPE_DEND,)):
        i = len(segs)
        seg = np.append(int(parentNode),
                        advance_segment(np.array(node, dtype=int), data,
                                        max_length=max_length, allowed_types=allowed_types))
        segs.append(seg)
        cons.append([parentSegment, i])

        last_node = int(segs[-1][-1])
        child = get_child(last_node, data, allowed_types=allowed_types).tolist()
        if len(child) > 0:
            segs, cons = get_segments([child[0]], last_node, i, segs, cons, data,
                                      max_length=max_length, allowed_types=allowed_types)
            for j in range(1, len(child)):
                segs, cons = get_segments([child[j]], last_node, i, segs, cons, data,
                                          max_length=max_length, allowed_types=allowed_types)
        return segs, cons

    def morph2Segs(swc_path, segment_types=(TYPE_DEND,), max_length=20.0):
        data = np.loadtxt(swc_path)
        if data.ndim != 2 or data.shape[1] < 7:
            raise ValueError(f"Invalid SWC table (expected at least seven columns): {swc_path}")

        ids = data[:, 0].astype(int)
        types = data[:, 1].astype(int)

        soma_ids = ids[types == TYPE_SOMA]
        if len(soma_ids) == 0:
            raise ValueError(f"SWC does not contain a soma node: {swc_path}")

        segs, cons = [], []
        for soma_id in soma_ids:
            children = get_child(int(soma_id), data, allowed_types=segment_types)
            for ch in children:
                segs, cons = get_segments([int(ch)], int(soma_id), -1, segs, cons, data,
                                          max_length=max_length, allowed_types=segment_types)

        if len(segs) == 0:
            return [], data

        soma_nodes = data[types == TYPE_SOMA]
        soma_mean = soma_nodes[:, 2:5].mean(axis=0)

        mp = id_to_row_index_map(data)
        seg_xy = []
        for seg in segs:
            sid, eid = int(seg[0]), int(seg[-1])
            p0 = data[mp[sid], 2:5] - soma_mean
            p1 = data[mp[eid], 2:5] - soma_mean
            seg_xy.append((float(p0[2]), float(p0[1]), float(p1[2]), float(p1[1])))  # z0,y0,z1,y1
        return seg_xy, data

    def load_swc_segments_zy(swc_path):
        seg_xy, _ = morph2Segs(swc_path, segment_types=(TYPE_DEND,), max_length=20.0)
        return np.asarray(seg_xy, dtype=float)

    swc_segments_by_layer = {}
    for layer in layers_of_interest:
        swc_path = layer_to_swc.get(layer, None)
        if (swc_path is None) or (not os.path.isfile(swc_path)):
            print(f"[WARN] Missing SWC for {layer}: {swc_path}")
            continue
        if layer not in layer_centers:
            print(f"[WARN] No patch-derived center for {layer}; skipping its SWC.")
            continue
        swc_segments_by_layer[layer] = load_swc_segments_zy(swc_path)
        print(f"[SWC] {layer}: segments={len(swc_segments_by_layer[layer])}")

    # ======================================================
    # ======================================================
    def prepare_syn(df, pre_col, post_col, syn_id_col, tag):
        d = df.copy()
        d = ensure_post_xyz_transformed(d)

        d = d[d[post_col].isin(core_ids)].copy()

        # post_class/post_layer
        if "post_class" not in d.columns:
            tmp = cells_lookup[["pt_root_id", "classification_system_auto"]].rename(
                columns={"pt_root_id": "__pid", "classification_system_auto": "post_class"}
            )
            d = d.merge(tmp, left_on=post_col, right_on="__pid", how="left").drop(columns=["__pid"])

        if "post_layer" not in d.columns:
            tmp = cells_lookup[["pt_root_id", "layer_auto"]].rename(
                columns={"pt_root_id": "__pid", "layer_auto": "post_layer"}
            )
            d = d.merge(tmp, left_on=post_col, right_on="__pid", how="left").drop(columns=["__pid"])

        # post soma y/z
        tmp = cells_lookup[["pt_root_id", "pt_yt", "pt_zt"]].rename(
            columns={"pt_root_id": "__pid", "pt_yt": "post_soma_yt", "pt_zt": "post_soma_zt"}
        )
        d = d.merge(tmp, left_on=post_col, right_on="__pid", how="left").drop(columns=["__pid"])

        d = d[d["post_class"] == "excitatory_neuron"].copy()
        d = d[d["post_layer"].isin(layers_of_interest)].copy()

        # synapse_id
        if syn_id_col is None:
            d["synapse_id"] = (tag + "_") + d.index.astype(str)
        else:
            d["synapse_id"] = d[syn_id_col].astype(str)

        # Synapse-processing details.
        use_syn = False
        if all(c in d.columns for c in ["post_yt", "post_zt"]) and d[["post_yt", "post_zt"]].notna().all(axis=1).any():
            good = d[["post_yt", "post_zt"]].notna().all(axis=1)
            d.loc[~good, "post_yt"] = d.loc[~good, "post_soma_yt"]
            d.loc[~good, "post_zt"] = d.loc[~good, "post_soma_zt"]
            use_syn = True
        else:
            d["post_yt"] = d["post_soma_yt"]
            d["post_zt"] = d["post_soma_zt"]

        d["dy"] = d["post_yt"].astype(float) - d["post_soma_yt"].astype(float)
        d["dy_abs"] = d["dy"].abs()
        return d, use_syn

    syn_ee_u, ee_used_syn = prepare_syn(syn_ee, pre_col_ee, post_col_ee, syn_id_col_ee, "EE")
    syn_ie_u, ie_used_syn = prepare_syn(syn_ie, pre_col_ie, post_col_ie, syn_id_col_ie, "IE")
    syn_all = pd.concat([syn_ee_u, syn_ie_u], ignore_index=True)

    ee_coordinate_source = "synapse coordinates" if ee_used_syn else "postsynaptic soma coordinates"
    ie_coordinate_source = "synapse coordinates" if ie_used_syn else "postsynaptic soma coordinates"
    print(f"[INFO] EE postsynaptic coordinate source: {ee_coordinate_source}")
    print(f"[INFO] IE postsynaptic coordinate source: {ie_coordinate_source}")
    print("[INFO] total syn used:", len(syn_all))

    layer_q90_pos, layer_q90_neg = {}, {}
    for layer in layers_of_interest:
        sub = syn_all[syn_all["post_layer"] == layer]
        if len(sub) == 0:
            layer_q90_pos[layer] = None
            layer_q90_neg[layer] = None
            continue
        sub_pos = sub[sub["dy"] > 0]
        sub_neg = sub[sub["dy"] < 0]
        layer_q90_pos[layer] = float(sub_pos["dy_abs"].quantile(0.9)) if len(sub_pos) else None
        layer_q90_neg[layer] = float(sub_neg["dy_abs"].quantile(0.9)) if len(sub_neg) else None

    # ======================================================
    # ======================================================
    def map_layer_syn_to_segment(layer: str, df_layer: pd.DataFrame) -> pd.DataFrame:
        if layer not in swc_segments_by_layer:
            return pd.DataFrame(columns=["synapse_id", "segment_id"])
        if layer not in layer_centers:
            return pd.DataFrame(columns=["synapse_id", "segment_id"])

        segs = swc_segments_by_layer[layer]  # (nseg,4) z0,y0,z1,y1 REL
        nseg = segs.shape[0]
        if nseg == 0:
            return pd.DataFrame(columns=["synapse_id", "segment_id"])

        y_c = layer_centers[layer]["y"]

        dy = df_layer["dy"].to_numpy(dtype=float)
        dy_abs = df_layer["dy_abs"].to_numpy(dtype=float)

        y0_rel = segs[:, 1]
        y1_rel = segs[:, 3]
        ys = np.concatenate([y0_rel, y1_rel])

        pos_span = float(np.max(ys[ys >= 0])) if np.any(ys >= 0) else 20.0
        neg_span = float(abs(np.min(ys[ys <= 0]))) if np.any(ys <= 0) else 20.0
        if pos_span <= 0: pos_span = 20.0
        if neg_span <= 0: neg_span = 20.0

        q90_pos = layer_q90_pos.get(layer) or 1.0
        q90_neg = layer_q90_neg.get(layer) or 1.0
        if q90_pos <= 0: q90_pos = 1.0
        if q90_neg <= 0: q90_neg = 1.0

        rel = np.zeros_like(dy_abs)
        pos_idx = dy > 0
        neg_idx = dy < 0
        if np.any(pos_idx):
            rel[pos_idx] = np.clip(dy_abs[pos_idx] / q90_pos, 0.0, 1.0)
        if np.any(neg_idx):
            rel[neg_idx] = np.clip(dy_abs[neg_idx] / q90_neg, 0.0, 1.0)

        y_rel = np.zeros_like(dy, dtype=float)
        y_rel[pos_idx] = rel[pos_idx] * pos_span
        y_rel[neg_idx] = -rel[neg_idx] * neg_span

        y_target_plot = (y_c + y_rel)

        seg_y_mid_plot = ((0.5 * (y0_rel + y1_rel) + y_c) )
        seg_y_min_plot = ((np.minimum(y0_rel, y1_rel) + y_c))
        seg_y_max_plot = ((np.maximum(y0_rel, y1_rel) + y_c))

        tol_y = 2.0
        seg_id = np.zeros(len(df_layer), dtype=int)

        for i, yt in enumerate(y_target_plot):
            mask = (yt >= (seg_y_min_plot - tol_y)) & (yt <= (seg_y_max_plot + tol_y))
            if np.any(mask):
                cand = np.where(mask)[0]
                mid = seg_y_mid_plot[cand]
                seg_id[i] = int(cand[np.argmin(np.abs(mid - yt))])
            else:
                seg_id[i] = int(np.argmin(np.abs(seg_y_mid_plot - yt)))

        return pd.DataFrame({
            "synapse_id": df_layer["synapse_id"].astype(str).to_numpy(),
            "segment_id": seg_id.astype(int),
        })

    # ======================================================
    # Output handling.
    # ======================================================
    layer_to_outname = {
        "L2/3": "syn_to_segment_L2_3.csv",
        "L4":   "syn_to_segment_L4.csv",
        "L5":   "syn_to_segment_L5.csv",
        "L6":   "syn_to_segment_L6.csv",
    }

    out_paths = {}
    for layer in layers_of_interest:
        sub = syn_all[syn_all["post_layer"] == layer].copy()
        if len(sub) == 0:
            print(f"[SKIP] {layer}: no eligible synapses")
            continue
        if layer not in swc_segments_by_layer:
            print(f"[SKIP] {layer}: no valid SWC segments")
            continue

        m = map_layer_syn_to_segment(layer, sub)
        out_path = os.path.join(out_dir, layer_to_outname[layer])
        m.to_csv(out_path, index=False, encoding="utf-8-sig")
        out_paths[layer] = out_path
        print(f"[SAVE] {layer}: {len(m)} rows -> {out_path}")

    print("[COMPLETE] synapse-to-segment maps:", out_dir)
    return out_paths


