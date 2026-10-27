# -*- coding: utf-8 -*-
"""生成主要的100个配对MICrONS与ER动力学试次。 / Generate the primary 100 paired MICrONS and ER dynamics trials."""

import os
import argparse
import random
import heapq
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from standard_transform import minnie_ds


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("MICRONS_DATA_ROOT", REPOSITORY_ROOT / "data" / "microns")
).resolve()
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")
).resolve()
ALLEN_ROOT = PROJECT_ROOT / "ALLEN"

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# =========================================================
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


# =========================================================
# 返回值细节。 / Return-value details.
# =========================================================
def simulate_event_driven_branching(
    G,
    rho,
    tmax_sec=20.0,
    dt=0.001,
    m0=0.99,
    refractory_period=0.001,
    external_rate=0.01,
    syn_delay_mean=0.001,
    warmup_time=2.0,
    name="network",
    seed=None,
    MAX_QUEUE=2_000_000
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    successors = {u: list(G.successors(u)) for u in G.nodes()}
    external_nodes = list(G.nodes())

    out_weights = {}
    out_strength = {}
    for u in G.nodes():
        nbrs = successors[u]
        if len(nbrs) == 0:
            out_weights[u] = np.array([], dtype=float)
            out_strength[u] = 0.0
            continue

        w_list = []
        for v in nbrs:
            w = float(G[u][v].get("weight", 1.0))
            if w < 0:
                w = 0.0
            w_list.append(w)

        w_arr = np.asarray(w_list, dtype=float)
        out_weights[u] = w_arr

        s = float(np.sum(w_arr))
        if not np.isfinite(s):
            s = 0.0
        out_strength[u] = s

    event_queue = []
    heapq.heapify(event_queue)

    next_available_time = {node: 0.0 for node in G.nodes()}

    spike_times = []

    # 统计分析。 / Statistical analysis.
    total_internal_parent_spikes_enqueue = 0
    total_internal_children_enqueued = 0

    # 统计分析。 / Statistical analysis.
    internal_parent_spikes_actual = 0
    internal_child_spikes_actual  = 0

    # 输入处理。 / Input handling.
    t_full = np.arange(0, tmax_sec, dt)
    lam = external_rate * dt * 1324 * (1 - m0)
    exN_full = np.random.poisson(lam=lam, size=len(t_full))
    ext_total=int(exN_full.sum())
    print(f"[{name}] external events drawn: {int(exN_full.sum())}")

    for t_now, n_input in zip(t_full, exN_full):
        if n_input > 0:
            chosen = random.choices(external_nodes, k=int(n_input))
            for nd in chosen:
                heapq.heappush(event_queue, (float(t_now), "external_spike", int(nd), {"src": "external"}))

    if rho <= 0:
        raise ValueError(f"[{name}] spectral radius must be positive; received rho={rho}.")
    g_branch = m0 / rho
    print(f"[{name}] branching gain g=m0/rho={g_branch:.8f} (m0={m0}, rho={rho:.6f})")

    def _sample_children_posts(node, out_nbrs, n_children):
        if n_children <= 0:
            return []

        w_arr = out_weights[node]

        if w_arr.size != len(out_nbrs):
            return [random.choice(out_nbrs) for _ in range(int(n_children))]

        s_w = float(np.sum(w_arr))
        if (not np.isfinite(s_w)) or (s_w <= 0.0):
            return [random.choice(out_nbrs) for _ in range(int(n_children))]

        probs = w_arr / s_w
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.where(probs > 0.0, probs, 0.0)

        ps = float(np.sum(probs))
        if (not np.isfinite(ps)) or (ps <= 0.0):
            return [random.choice(out_nbrs) for _ in range(int(n_children))]

        probs = probs / ps
        try:
            return list(np.random.choice(out_nbrs, size=int(n_children), p=probs, replace=True))
        except Exception:
            return [random.choice(out_nbrs) for _ in range(int(n_children))]

    while event_queue:
        if len(event_queue) > MAX_QUEUE:
            print(f"[{name}] event queue exceeded {MAX_QUEUE:,} entries; terminating this trial.")
            break

        t_event, etype, node, payload = heapq.heappop(event_queue)
        if t_event > tmax_sec:
            break

        if etype == "external_spike":
            if t_event >= next_available_time[node]:
                spike_times.append((t_event, node))
                next_available_time[node] = t_event + refractory_period

                out_nbrs = successors[node]
                s = out_strength[node]
                if out_nbrs and s > 0:
                    mean_children = g_branch * s
                    n_children = np.random.poisson(mean_children)
                    if n_children > 0:
                        chosen_posts = _sample_children_posts(node, out_nbrs, n_children)
                        for post in chosen_posts:
                            delay = random.expovariate(1.0 / syn_delay_mean)
                            t_post = t_event + delay
                            post_i = int(post)
                            if t_post >= next_available_time[post_i]:
                                heapq.heappush(event_queue, (t_post, "node_spike", post_i, {"src": "external"}))

        elif etype == "node_spike":
            if t_event >= next_available_time[node]:
                spike_times.append((t_event, node))
                next_available_time[node] = t_event + refractory_period

                # 统计分析。 / Statistical analysis.
                src = payload.get("src", None) if isinstance(payload, dict) else None
                if src == "internal":
                    internal_child_spikes_actual += 1

                out_nbrs = successors[node]
                s = out_strength[node]
                if out_nbrs and s > 0:
                    internal_parent_spikes_actual += 1

                    total_internal_parent_spikes_enqueue += 1

                    mean_children = g_branch * s
                    n_children = np.random.poisson(mean_children)

                    if n_children > 0:
                        chosen_posts = _sample_children_posts(node, out_nbrs, n_children)

                        for post in chosen_posts:
                            delay = random.expovariate(1.0 / syn_delay_mean)
                            t_post = t_event + delay
                            post_i = int(post)

                            if t_post >= next_available_time[post_i]:
                                heapq.heappush(event_queue, (t_post, "node_spike", post_i, {"src": "internal"}))
                                total_internal_children_enqueued += 1

    spike_times = [(t, n) for (t, n) in spike_times if t >= warmup_time]

    if total_internal_parent_spikes_enqueue > 0:
        m_enqueue = total_internal_children_enqueued / total_internal_parent_spikes_enqueue
    else:
        m_enqueue = 0.0

    if internal_parent_spikes_actual > 0:
        m_actual = internal_child_spikes_actual / internal_parent_spikes_actual
    else:
        m_actual = 0.0

    print(f"[{name}][enqueue estimate] internal_parent={total_internal_parent_spikes_enqueue}  "
          f"child_enqueued={total_internal_children_enqueued}  m_enqueue={m_enqueue:.8f}")
    print(f"[{name}][realized estimate] internal_parent={internal_parent_spikes_actual}  "
          f"internal_child_spikes={internal_child_spikes_actual}  m_actual={m_actual:.8f}")

    print(f"[{name}] spikes retained after warm-up: {len(spike_times)}")
    return spike_times, float(m_actual),int(ext_total)



# =========================================================
# =========================================================
def compute_branching_series(spike_times, bin_size=0.001, warmup_time=2.0):
    t_all = [t for t, _ in spike_times if t >= warmup_time]
    if len(t_all) == 0:
        return np.array([]), np.array([])
    t_start, t_end = min(t_all), max(t_all)
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    B_t, _ = np.histogram(t_all, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, B_t


def bin_counts_E(spike_times_E, dt, t_start, t_end, N_E):
    n_bins = int(np.floor((t_end - t_start) / dt))
    E_total = np.zeros(n_bins, dtype=float)

    for t, node in spike_times_E:
        k = int((t - t_start) / dt)
        if 0 <= k < n_bins and (0 <= node < N_E):
            E_total[k] += 1.0

    t_centers = t_start + (np.arange(n_bins) + 0.5) * dt
    return t_centers, E_total


# =========================================================
# 统计分析。 / Statistical analysis.
# =========================================================
def bin_spikes_per_node(spike_times, N, dt, t_start, t_end):
    n_bins = int(np.floor((t_end - t_start) / dt))
    B = np.zeros((n_bins, N), dtype=float)

    for t, n in spike_times:
        k = int((t - t_start) / dt)
        if 0 <= k < n_bins and (0 <= n < N):
            B[k, n] += 1.0

    time_centers = t_start + (np.arange(n_bins) + 0.5) * dt
    return time_centers, B


# =========================================================
# 构建任务输入。 / Build task inputs.
# =========================================================
def build_microns_EE_EI_IE_matrices():
    celltype_path = DATA_ROOT / "proofread_cells_celltypes_auto.csv"
    meta_path = DATA_ROOT / "proofread_cells_metadata.csv"
    syn_path = DATA_ROOT / "proofread_cells_synapses.csv"

    df_ct   = pd.read_csv(celltype_path)
    df_meta = pd.read_csv(meta_path)
    df_syn  = pd.read_csv(syn_path)

    cells = df_meta.merge(df_ct, on="pt_root_id", how="left")

    XYZ_t = minnie_ds.transform_nm.apply_dataframe('pt_position', cells)
    XYZ_t = np.array(XYZ_t)
    cells['pt_xt'] = XYZ_t[:, 0]
    cells['pt_yt'] = XYZ_t[:, 1]
    cells['pt_zt'] = XYZ_t[:, 2]

    cells_pos = (
        cells[["pt_root_id", "pt_xt", "pt_yt", "pt_zt"]]
        .drop_duplicates("pt_root_id")
        .set_index("pt_root_id")
    )

    type_map = (
        df_ct[["pt_root_id", "classification_system_auto"]]
        .drop_duplicates("pt_root_id")
        .set_index("pt_root_id")
    )

    syn = df_syn.merge(
        type_map.rename(columns={"classification_system_auto": "pre_class"}),
        how="left", left_on="pre_pt_root_id", right_index=True,
    )
    syn = syn.merge(
        type_map.rename(columns={"classification_system_auto": "post_class"}),
        how="left", left_on="post_pt_root_id", right_index=True,
    )

    mask_spatial = (
        (cells_pos["pt_xt"] >= 550.0) &
        (cells_pos["pt_xt"] <= 800.0) &
        (cells_pos["pt_zt"] >= 750.0) &
        (cells_pos["pt_zt"] <= 1100.0)
    )
    region_ids = set(cells_pos[mask_spatial].index.to_numpy())
    if len(region_ids) == 0:
        raise RuntimeError("No MICrONS cells fall inside the configured spatial patch.")

    # 兴奋到兴奋连接。 / ---------------- E→E ----------------
    syn_ee = syn[
        (syn["pre_class"] == "excitatory_neuron") &
        (syn["post_class"] == "excitatory_neuron") &
        (syn["pre_pt_root_id"].isin(region_ids)) &
        (syn["post_pt_root_id"].isin(region_ids))
    ].copy()
    if len(syn_ee) == 0:
        raise RuntimeError("No E-to-E synapses remain after spatial and cell-type filtering.")

    syn_ee["weight"] = 1.0

    ee_ids = pd.unique(
        np.concatenate([syn_ee["pre_pt_root_id"].to_numpy(),
                        syn_ee["post_pt_root_id"].to_numpy()])
    )
    N_E = len(ee_ids)
    e_id_to_idx = {rid: i for i, rid in enumerate(ee_ids)}

    W_EE = np.zeros((N_E, N_E), dtype=float)
    for row in syn_ee.itertuples(index=False):
        i = e_id_to_idx[row.post_pt_root_id]  # 行表示突触后节点。 / post row
        j = e_id_to_idx[row.pre_pt_root_id]   # 列表示突触前节点。 / pre col
        W_EE[i, j] += row.weight

    # 兴奋到抑制连接。 / ---------------- E→I ----------------
    syn_ei = syn[
        (syn["pre_class"] == "excitatory_neuron") &
        (syn["post_class"] == "inhibitory_neuron") &
        (syn["pre_pt_root_id"].isin(region_ids)) &
        (syn["post_pt_root_id"].isin(region_ids))
    ].copy()

    syn_ie = syn[
        (syn["pre_class"] == "inhibitory_neuron") &
        (syn["post_class"] == "excitatory_neuron") &
        (syn["pre_pt_root_id"].isin(region_ids)) &
        (syn["post_pt_root_id"].isin(region_ids))
    ].copy()

    i_ids = pd.unique(
        np.concatenate([
            syn_ei["post_pt_root_id"].to_numpy() if len(syn_ei) else np.array([], dtype=np.int64),
            syn_ie["pre_pt_root_id"].to_numpy()  if len(syn_ie) else np.array([], dtype=np.int64),
        ])
    )
    i_ids = i_ids.astype(np.int64)
    N_I = len(i_ids)

    i_id_to_idx = {rid: i for i, rid in enumerate(i_ids)}

    if len(syn_ei) == 0 or N_I == 0:
        W_EI = np.zeros((0, N_E), dtype=float)
    else:
        syn_ei["weight"] = 1.0
        W_EI = np.zeros((N_I, N_E), dtype=float)
        for row in syn_ei.itertuples(index=False):
            pre_id = row.pre_pt_root_id   # 兴奋性。 / E
            post_id = row.post_pt_root_id # 抑制性。 / I
            if pre_id not in e_id_to_idx or post_id not in i_id_to_idx:
                continue
            i = i_id_to_idx[post_id]
            j = e_id_to_idx[pre_id]
            W_EI[i, j] += row.weight

    if len(syn_ie) == 0 or N_I == 0:
        W_IE = np.zeros((N_E, N_I), dtype=float)
    else:
        syn_ie["weight"] = 1.0
        W_IE = np.zeros((N_E, N_I), dtype=float)
        for row in syn_ie.itertuples(index=False):
            pre_id = row.pre_pt_root_id   # 抑制性。 / I
            post_id = row.post_pt_root_id # 兴奋性。 / E
            if post_id not in e_id_to_idx or pre_id not in i_id_to_idx:
                continue
            e = e_id_to_idx[post_id]
            i = i_id_to_idx[pre_id]
            W_IE[e, i] += row.weight

    # 统计分析。 / Statistical analysis.
    # 已注释的诊断输出：兴奋细胞数、兴奋到兴奋边数及权重总和。 / print("N_E =", N_E, "  E→E edges =", int(np.count_nonzero(W_EE)), "  sum(E→E) =", int(W_EE[W_EE > 0].sum()))
    # 已注释的诊断输出：抑制细胞数、兴奋到抑制边数及权重总和。 / print("N_I =", N_I, "  E→I edges =", int(np.count_nonzero(W_EI)), "  sum(E→I) =", int(W_EI[W_EI > 0].sum()) if W_EI.size else 0)
    # 已注释的诊断输出：抑制细胞数、抑制到兴奋边数及权重总和。 / print("N_I =", N_I, "  I→E edges =", int(np.count_nonzero(W_IE)), "  sum(I→E) =", int(W_IE[W_IE > 0].sum()) if W_IE.size else 0)

    return W_EE, W_EI, W_IE, ee_ids, i_ids


# =========================================================
# 突触处理细节。 / Synapse-processing details.
# =========================================================
def build_ER_control_uniform_synapses(W_real, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = W_real.shape[0]
    real_edges = int(np.count_nonzero(W_real))
    if real_edges == 0:
        raise RuntimeError("The empirical adjacency matrix contains no nonzero edges.")

    total_syn_real = int(np.round(W_real[W_real > 0].sum()))
    if total_syn_real <= 0:
        raise RuntimeError("The empirical adjacency matrix has zero total synaptic weight.")

    p_er = real_edges / (N * N)
    ER_mask = (np.random.rand(N, N) < p_er)
    np.fill_diagonal(ER_mask, 0)

    idx = np.argwhere(ER_mask)
    n_er_edges = idx.shape[0]

    if n_er_edges == 0:
        all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        chosen = np.random.choice(len(all_pairs), size=real_edges, replace=False)
        idx = np.array([all_pairs[k] for k in chosen], dtype=int)
        n_er_edges = idx.shape[0]

    if n_er_edges > real_edges:
        pick = np.random.choice(n_er_edges, size=real_edges, replace=False)
        idx = idx[pick]
        n_er_edges = real_edges
    elif n_er_edges < real_edges:
        need = real_edges - n_er_edges
        mask_now = np.zeros((N, N), dtype=bool)
        mask_now[idx[:, 0], idx[:, 1]] = True
        np.fill_diagonal(mask_now, True)
        cand = np.argwhere(~mask_now)
        add = cand[np.random.choice(cand.shape[0], size=need, replace=False)]
        idx = np.vstack([idx, add])
        n_er_edges = real_edges

    base = total_syn_real // n_er_edges
    rem  = total_syn_real - base * n_er_edges
    weights = np.full(n_er_edges, base, dtype=int)
    if rem > 0:
        plus_idx = np.random.choice(n_er_edges, size=rem, replace=False)
        weights[plus_idx] += 1

    if base == 0:
        weights[:] = 1
        print("[WARN] Fewer synapses than ER edges; assigning unit edge weights.")

    W_er = np.zeros((N, N), dtype=int)
    for (i, j), w in zip(idx, weights):
        W_er[i, j] = int(w)

    return W_er


# =========================================================
# 突触处理细节。 / Synapse-processing details.
# =========================================================
def build_ER_IE_uniform_synapses(N_I, N_E, num_edges, total_syn, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if num_edges <= 0 or total_syn <= 0 or N_I <= 0 or N_E <= 0:
        return np.zeros((N_E, N_I), dtype=int)

    max_edges = N_I * N_E
    if num_edges > max_edges:
        raise ValueError(f"num_edges ({num_edges}) exceeds N_I*N_E ({max_edges}).")

    # 可重复随机化细节。 / Reproducible randomization detail.
    all_pairs_idx = np.random.choice(max_edges, size=num_edges, replace=False)
    i_idx = (all_pairs_idx // N_E).astype(int)
    e_idx = (all_pairs_idx % N_E).astype(int)

    base = total_syn // num_edges
    rem  = total_syn - base * num_edges
    weights = np.full(num_edges, base, dtype=int)
    if rem > 0:
        plus = np.random.choice(num_edges, size=rem, replace=False)
        weights[plus] += 1

    if base == 0:
        weights[:] = 1
        print("[WARN] Fewer synapses than I-to-E edges; assigning unit edge weights.")

    W_IE = np.zeros((N_E, N_I), dtype=int)
    for ii, ee, w in zip(i_idx, e_idx, weights):
        W_IE[ee, ii] = int(w)

    return W_IE


# =========================================================
# =========================================================
def spectral_radius(W, name=""):
    eigvals = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
    print(f"[{name}] rho = {rho:.6f}")
    return rho, eigvals


def matrix_to_digraph(W):
    N = W.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    rows, cols = np.nonzero(W)
    for i, j in zip(rows, cols):
        G.add_edge(int(j), int(i), weight=float(W[i, j]))
    return G


# =========================================================
# =========================================================
def run_until_meff_close(
    G,
    rho_eff,
    m0,
    tol=1e-4,
    max_trials=200,
    base_seed=0,
    name="NET",
    sim_kwargs=None,
):
    if sim_kwargs is None:
        sim_kwargs = {}

    best_spikes = None
    best_meff = None
    best_abs_err = np.inf
    best_trial = -1
    best_seed = None
    best_ext_total=None

    for trial in range(1, max_trials + 1):
        seed = base_seed + trial

        spikes, meff,ext_total = simulate_event_driven_branching(
            G, rho=rho_eff, m0=m0, name=f"{name}_trial{trial}", seed=seed, **sim_kwargs
        )
        abs_err = abs(meff - m0) if meff > 0 else np.inf
        print(f"[{name}] Trial {trial}: m_eff={meff:.8f}, |Δ|={abs_err:.6e}")

        if abs_err < best_abs_err:
            best_abs_err = abs_err
            best_spikes = spikes
            best_meff = meff
            best_trial = trial
            best_seed = seed
            best_ext_total=int(ext_total)
        if abs_err < tol:
            print(f"[{name}] accepted inner trial {trial}: |m_eff-m0| < {tol}.")
            return spikes, meff, trial, seed,int(ext_total)

    print(f"[{name}] tolerance was not reached after {max_trials} inner trials; using best match: "
          f"Trial {best_trial}, m_eff={best_meff:.8f}, |Δ|={best_abs_err:.6e}")
    return best_spikes, best_meff, best_trial, best_seed,int(best_ext_total or 0)


# =========================================================
# =========================================================
def generate_ER_I_spikes_by_E_bins(
    E_bin_counts,
    N_I,
    dt,
    t_start,
    EI_ratio=1.0,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    I_spikes = []
    n_bins = len(E_bin_counts)

    for k in range(n_bins):
        lam = max(EI_ratio * float(E_bin_counts[k]), 0.0)
        if lam <= 0:
            continue
        nI = np.random.poisson(lam)
        if nI <= 0:
            continue

        # 神经元处理细节。 / Neuron-processing details.
        chosen_i = np.random.choice(N_I, size=int(nI), replace=True)

        t0 = t_start + k * dt
        for i_idx in chosen_i:
            t_evt = t0 + random.random() * dt
            I_spikes.append((t_evt, int(i_idx)))

    return I_spikes


# =========================================================
# 保存生成的输出。 / Save generated output.
# =========================================================
def save_spikes_ms_csv_from_dict(spikes_by_id, out_path):
    ensure_dir(out_path)

    if not spikes_by_id:
        print(f"[WARN] No spikes available; file was not written: {out_path}")
        return

    all_ids = sorted(spikes_by_id.keys())
    max_spikes = max(len(v) for v in spikes_by_id.values())

    rows = []
    for nid in all_ids:
        times = sorted(spikes_by_id[nid])
        times_ms = [f"{t*1000.0:.3f}" for t in times]
        row = [int(nid)] + times_ms + [""] * (max_spikes - len(times_ms))
        rows.append(row)

    cols = ["neuron_id"] + [f"spike_{i+1}" for i in range(max_spikes)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_path, index=False, header=False)
    print(f"[SAVED] spike table: {out_path}")


def build_spike_dict_from_spike_times(spike_times, id_list, id_offset=0):
    d = {}
    for t, idx in spike_times:
        nid = int(id_offset + int(id_list[idx]))
        d.setdefault(nid, []).append(float(t))
    return d


# =========================================================
# 保存生成的输出。 / Save generated output.
# =========================================================
def save_er_edges_weights_EE(W_er, out_path):
    ensure_dir(out_path)
    rows, cols = np.nonzero(W_er)
    pre = cols.astype(int)    # 突触前索引为j。 / pre=j
    post = rows.astype(int)   # 突触后索引为i。 / post=i
    w = W_er[rows, cols].astype(int)
    df = pd.DataFrame({"pre_id": pre, "post_id": post, "weight": w})
    df.to_csv(out_path, index=False)
    print(f"[SAVED] ER E-to-E edge table: {out_path} (n_edges={len(df)})")


def save_er_edges_weights_IE(W_IE_er, I_id_offset, out_path):
    ensure_dir(out_path)
    rows, cols = np.nonzero(W_IE_er)  # 行表示兴奋细胞，列表示抑制细胞。 / rows=E, cols=I
    post_E = rows.astype(int)
    pre_I = (cols.astype(int) + int(I_id_offset))
    w = W_IE_er[rows, cols].astype(int)

    df = pd.DataFrame({"pre_id": pre_I, "post_id": post_E, "weight": w})
    df.to_csv(out_path, index=False)
    print(f"[SAVED] ER I-to-E edge table: {out_path} (n_edges={len(df)})")

def run_one_trial(trial_idx: int, base_out_dir: str,
                  m0=0.99, tol=1e-3, max_trials=200,
                  tmax_sec=20.0, dt=0.001, warmup_time=2.0,
                  refractory_period=0.001, external_rate=0.1, syn_delay_mean=0.001,
                  Areal=1.19116, Aer=1.030544, EI_ratio=1.0,
                  show_plot=False):
    trial_dir = os.path.join(base_out_dir, f"ER_trial{trial_idx:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    W_EE_real, W_EI_real, W_IE_real, ee_ids, i_ids = build_microns_EE_EI_IE_matrices()

    real_w = W_EE_real[W_EE_real > 0].ravel()
    rho_real, _ = spectral_radius(W_EE_real, name=f"MICRONS_EE_trial{trial_idx:04d}")

    # 统计分析。 / Statistical analysis.
    N_E = W_EE_real.shape[0]
    N_I_real = W_IE_real.shape[1]
    num_IE_edges_real = int(np.count_nonzero(W_IE_real))
    total_IE_syn_real = int(W_IE_real[W_IE_real > 0].sum()) if W_IE_real.size else 0

    # 构建任务输入。 / Build task inputs.
    W_er = build_ER_control_uniform_synapses(W_EE_real, seed=10000 + trial_idx)
    er_w = W_er[W_er > 0].ravel()
    rho_er, _ = spectral_radius(W_er, name=f"ER_EE_trial{trial_idx:04d}")

    G_real = matrix_to_digraph(W_EE_real)
    G_er   = matrix_to_digraph(W_er)

    sim_kwargs = dict(
        tmax_sec=tmax_sec,
        dt=dt,
        warmup_time=warmup_time,
        refractory_period=refractory_period,
        external_rate=external_rate,
        syn_delay_mean=syn_delay_mean,
    )

    rho_eff_real = rho_real / Areal
    spikes_real_E, meff_real, trial_real, seed_real,ext_real = run_until_meff_close(
        G_real, rho_eff=rho_eff_real, m0=m0, tol=tol, max_trials=max_trials,
        base_seed=1001 + 1000*trial_idx,
        name=f"MICRONS_trial{trial_idx:04d}",
        sim_kwargs=sim_kwargs
    )

    rho_eff_er = rho_er / Aer
    spikes_er_E, meff_er, trial_er, seed_er,ext_er = run_until_meff_close(
        G_er, rho_eff=rho_eff_er, m0=m0, tol=tol, max_trials=max_trials,
        base_seed=2001 + 1000*trial_idx,
        name=f"ER_trial{trial_idx:04d}",
        sim_kwargs=sim_kwargs
    )

    # =====================================================
    # 构建任务输入。 / Build task inputs.
    # =====================================================
    if N_I_real <= 0:
        spikes_er_I = []
        W_IE_er = np.zeros((N_E, 0), dtype=int)
    else:
        # 统计分析。 / Statistical analysis.
        t_centers, E_bin = bin_counts_E(
            spikes_er_E, dt=dt, t_start=warmup_time, t_end=tmax_sec, N_E=N_E
        )

        spikes_er_I = generate_ER_I_spikes_by_E_bins(
            E_bin_counts=E_bin,
            N_I=N_I_real,
            dt=dt,
            t_start=warmup_time,
            EI_ratio=EI_ratio,        
            seed=3000 + trial_idx
        )

        # 构建任务输入。 / Build task inputs.
        W_IE_er = build_ER_IE_uniform_synapses(
            N_I=N_I_real,
            N_E=N_E,
            num_edges=num_IE_edges_real,
            total_syn=total_IE_syn_real,
            seed=4000 + trial_idx
        )

    # =====================================================
    # 输出处理。 / Output handling.
    # =====================================================
    t_real, B_real = compute_branching_series(spikes_real_E, bin_size=dt, warmup_time=warmup_time)
    t_er,   B_er   = compute_branching_series(spikes_er_E,   bin_size=dt, warmup_time=warmup_time)

    fig_path = os.path.join(trial_dir, "Bt_real_vs_er.png")
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        if len(t_real) > 0:
            axes[0].plot(t_real, B_real, lw=0.8)
            axes[0].set_ylabel("B(t)")
            axes[0].set_title(f"MICRONS E→E B(t)  (A={Areal:.6f}, m_eff={meff_real:.6f})")
            axes[0].grid(True)

        if len(t_er) > 0:
            axes[1].plot(t_er, B_er, lw=0.8)
            axes[1].set_xlabel("time (s)")
            axes[1].set_ylabel("B(t)")
            axes[1].set_title(f"ER(E) B(t)  (A={Aer:.6f}, m_eff={meff_er:.6f})")
            axes[1].grid(True)

        if len(t_real) > 0 and len(t_er) > 0:
            t_min = max(t_real.min(), t_er.min())
            t_max = min(t_real.max(), t_er.max())
            axes[0].set_xlim(t_min, t_max)

        plt.tight_layout()
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    except Exception as e:
        print(f"[trial {trial_idx:04d}] diagnostic plot failed: {e}")

    # =====================================================
    # 保存生成的输出。 / Save generated output.
    # =====================================================
    out_real_spk = os.path.join(trial_dir, "microns_patch_spikes_real_ms.csv")
    out_er_spk   = os.path.join(trial_dir, "microns_patch_spikes_ER_ms.csv")
    out_er_ee_w  = os.path.join(trial_dir, "microns_patch_edges_ER_weights.csv")
    out_er_ie_w  = os.path.join(trial_dir, "microns_patch_edges_ER_IE_weights.csv")

    # =====================================================
    # 保存生成的输出。 / Save generated output.
    # =====================================================
    neuron_spikes_real = {}  # 键为pt_root_id，值为秒单位的时间列表。 / key: pt_root_id, value: list[t_sec]

    for t_sec, node_idx in spikes_real_E:
        pt_id = int(ee_ids[int(node_idx)])
        neuron_spikes_real.setdefault(pt_id, []).append(float(t_sec))

    if W_EI_real is not None and W_EI_real.size > 0 and len(i_ids) > 0:
        # 统计分析。 / Statistical analysis.
        t_E, B_E_node = bin_spikes_per_node(
            spikes_real_E,
            N=N_E,
            dt=dt,
            t_start=warmup_time,
            t_end=tmax_sec
        )

        # 按兴奋到抑制权重对兴奋节点活动求和，得到抑制输入。 / raw_I[t, i] = sum_j B_E_node[t, j] * W_EI[i, j]
        raw_I = B_E_node @ W_EI_real.T   # 形状为时间点数乘以抑制细胞数。 / (T, N_I)



        E_sum_t = B_E_node.sum(axis=1)      # 形状为时间点数的一维数组。 / (T,)
        I_raw_sum_t = raw_I.sum(axis=1)     # 形状为时间点数的一维数组。 / (T,)

        scale_t = np.zeros_like(E_sum_t, dtype=float)
        mask_nonzero = I_raw_sum_t > 0
        scale_t[mask_nonzero] = (EI_ratio * E_sum_t[mask_nonzero]) / I_raw_sum_t[mask_nonzero]

        I_scaled = raw_I * scale_t[:, None]  # 形状为时间点数乘以抑制细胞数。 / (T, N_I)

        rng = np.random.default_rng(5000 + trial_idx)  # 可重复随机化细节。 / Reproducible randomization detail.
        T_bins, N_I_tmp = I_scaled.shape
        if N_I_tmp == len(i_ids):
            for kbin in range(T_bins):
                t_center = float(t_E[kbin])
                t_start_bin = t_center - 0.5 * dt
                for i_idx in range(N_I_tmp):
                    lam_bin = float(I_scaled[kbin, i_idx])
                    if (not np.isfinite(lam_bin)) or (lam_bin <= 0.0):
                        continue
                    n_sp = int(rng.poisson(lam_bin))
                    if n_sp <= 0:
                        continue
                    pt_id_I = int(i_ids[int(i_idx)])
                    for _ in range(n_sp):
                        t_evt = t_start_bin + float(rng.random()) * dt
                        neuron_spikes_real.setdefault(pt_id_I, []).append(float(t_evt))
        else:
            print(
                f"[trial {trial_idx:04d}] inhibitory activity shape does not match i_ids; "
                "REAL inhibitory spikes were not generated."
            )
    else:
        pass

    # 保存生成的输出。 / Save generated output.
    save_spikes_ms_csv_from_dict(neuron_spikes_real, out_real_spk)

    I_id_offset = N_E
    er_ids_E = np.arange(N_E, dtype=int)
    er_dict_E = build_spike_dict_from_spike_times(spikes_er_E, id_list=er_ids_E, id_offset=0)

    er_ids_I = np.arange(N_I_real, dtype=int)
    er_dict_I = build_spike_dict_from_spike_times(spikes_er_I, id_list=er_ids_I, id_offset=I_id_offset)

    spikes_er_dict = {}
    for d in (er_dict_E, er_dict_I):
        for k, v in d.items():
            spikes_er_dict.setdefault(k, []).extend(v)
    save_spikes_ms_csv_from_dict(spikes_er_dict, out_er_spk)

    save_er_edges_weights_EE(W_er, out_er_ee_w)

    if N_I_real > 0 and int(np.count_nonzero(W_IE_er)) > 0:
        save_er_edges_weights_IE(W_IE_er, I_id_offset=I_id_offset, out_path=out_er_ie_w)
    else:
        print(f"[trial {trial_idx:04d}] no ER I-to-E edges; table not written: {out_er_ie_w}")

    # =====================================================
    # 保存生成的输出。 / Save generated output.
    # =====================================================
    summary_path = os.path.join(trial_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("======= Trial Summary =======\n")
        f.write(f"trial_idx: {trial_idx}\n")
        f.write(f"EI_ratio: {EI_ratio}\n")
        f.write(f"m0: {m0}\n")
        f.write(f"tol: {tol}\n")
        f.write(f"tmax_sec: {tmax_sec}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"warmup_time: {warmup_time}\n")
        f.write(f"external_rate: {external_rate}\n")
        f.write(f"syn_delay_mean: {syn_delay_mean}\n")
        f.write(f"refractory_period: {refractory_period}\n")
        f.write("\n--- Real (MICRONS E) ---\n")
        f.write(f"external_inputs_real_selected: {int(ext_real)}\n")
        f.write(f"Areal: {Areal}\n")
        f.write(f"rho_real: {rho_real}\n")
        f.write(f"rho_eff_real: {rho_eff_real}\n")
        f.write(f"meff_real: {meff_real}\n")
        f.write(f"trial_real_inner: {trial_real}\n")
        f.write(f"seed_real: {seed_real}\n")
        f.write("\n--- ER (E) ---\n")
        f.write(f"external_inputs_er_selected: {int(ext_er)}\n")
        f.write(f"Aer: {Aer}\n")
        f.write(f"rho_er: {rho_er}\n")
        f.write(f"rho_eff_er: {rho_eff_er}\n")
        f.write(f"meff_er: {meff_er}\n")
        f.write(f"trial_er_inner: {trial_er}\n")
        f.write(f"seed_er: {seed_er}\n")
        f.write("\n--- I->E matching ---\n")
        f.write(f"N_I_real: {N_I_real}\n")
        f.write(f"num_IE_edges_real: {num_IE_edges_real}\n")
        f.write(f"total_IE_syn_real: {total_IE_syn_real}\n")
        f.write(f"ER_I_spikes: {len(spikes_er_I)}\n")
        f.write(f"W_IE_er_nonzero_edges: {int(np.count_nonzero(W_IE_er))}\n")
        f.write(f"W_IE_er_total_syn: {int(W_IE_er[W_IE_er>0].sum()) if W_IE_er.size else 0}\n")

    print(f"\n[trial {trial_idx:04d}] completed -> {trial_dir}")
    return dict(
        trial_idx=int(trial_idx),
        trial_dir=str(trial_dir),
        EI_ratio=float(EI_ratio),     
        meff_real=float(meff_real),
        meff_er=float(meff_er),
        nsp_real=int(len(spikes_real_E)),
        nsp_er_E=int(len(spikes_er_E)),
        nsp_er_I=int(len(spikes_er_I)),
        rho_real=float(rho_real),
        rho_er=float(rho_er),
        ext_real=int(ext_real),
        ext_er=int(ext_er),
    )



def _worker_run_one_trial(args):
    return run_one_trial(**args)


import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed



def _worker_run_one_trial(kwargs):
    return run_one_trial(**kwargs)


import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# 已注释的单试次工作函数定义。 / def _worker_run_one_trial(kwargs):
# 已注释的调用：展开关键字参数并返回单试次结果。 / return run_one_trial(**kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the 100 m=0.99 MICrONS and ER dynamics trials."
    )
    parser.add_argument("--output-root", type=Path, default=ALLEN_ROOT / "file" / "final_trial")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--m", type=float, default=0.99)
    args = parser.parse_args()

    out_root = str(args.output_root.resolve())
    os.makedirs(out_root, exist_ok=True)

    N_RUNS = int(args.n_runs)
    EI_min = 1.0 / 3.0  # 1:3
    EI_max = 3.0  # 3:1

    EI_ratios = np.linspace(EI_min, EI_max, N_RUNS, dtype=float)

    # 配置。 / Configuration.
    USE_THREADS = False
    max_workers = int(args.max_workers)
    Executor = ThreadPoolExecutor if USE_THREADS else ProcessPoolExecutor

    results = [None] * N_RUNS
    completed = 0

    partial_path = os.path.join(out_root, f"trial_summary_{N_RUNS}runs_partial.csv")
    final_path = os.path.join(out_root, f"trial_summary_{N_RUNS}runs.csv")

    futures = []
    with Executor(max_workers=max_workers) as ex:
        for k in range(1, N_RUNS + 1):
            ei_local = float(EI_ratios[k - 1])

            kwargs = dict(
                trial_idx=int(k),
                base_out_dir=out_root,
                m0=float(args.m),
                tol=5e-4,
                max_trials=200,
                tmax_sec=22.0,
                dt=0.001,
                warmup_time=2.0,
                refractory_period=0.001,
                external_rate=1,
                syn_delay_mean=0.001,
                Areal=1.19116,
                Aer=1.030544,
                EI_ratio=ei_local,
                # 返回值细节。 / Return-value details.
                show_plot=False
            )

            futures.append(ex.submit(_worker_run_one_trial, kwargs))

        for fut in as_completed(futures):
            row = fut.result()

            if isinstance(row, dict) and "trial_idx" in row:
                idx = int(row["trial_idx"]) - 1
                if 0 <= idx < N_RUNS:
                    results[idx] = row
            else:
                for j in range(N_RUNS):
                    if results[j] is None:
                        results[j] = row
                        break

            completed += 1

            if (completed % 50) == 0:
                df_tmp = pd.DataFrame([r for r in results if r is not None])
                if "trial_idx" in df_tmp.columns:
                    df_tmp = df_tmp.sort_values("trial_idx")
                df_tmp.to_csv(partial_path, index=False, encoding="utf-8-sig")
                print(f"[partial] {completed}/{N_RUNS} saved -> {partial_path}")

    df = pd.DataFrame([r for r in results if r is not None])
    if "trial_idx" in df.columns:
        df = df.sort_values("trial_idx")
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"\n[COMPLETE] trial summary -> {final_path}")
