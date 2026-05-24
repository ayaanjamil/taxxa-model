"""
Modal sharded GPU embedding of the Finnish tax corpus with BAAI/bge-m3 (1024-dim).

Splits 376k chunks across N parallel A10G workers. Each shard is short enough
(~10 min) that preemption is rare; if one is preempted, only that shard restarts.

Run:
    modal run modal_embed.py --shards 4

Live progress streams to your terminal. After all shards complete, the local
entrypoint concatenates them and writes data/vectors_bgem3.npy +
data/id_map_bgem3.json.
"""

import json
import sys
import time
from pathlib import Path

import modal

APP_NAME = "taxxa-bgem3-embed"
MODEL_NAME = "BAAI/bge-m3"
GPU = "A10G"
BATCH_SIZE = 192
MAX_LENGTH = 384  # p99 of our chunks ~331 tokens; 384 has headroom

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.45.2",
        "sentence-transformers==3.2.1",
        "numpy<2",
        "tqdm",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("taxxa-embed", create_if_missing=True)
model_cache = modal.Volume.from_name("taxxa-hf-cache", create_if_missing=True)


@app.function(
    gpu=GPU,
    volumes={"/data": volume, "/root/.cache/huggingface": model_cache},
    timeout=60 * 45,  # 45-min cap per shard
)
def embed_shard(shard_index: int, n_shards: int) -> dict:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    tag = f"[s{shard_index}/{n_shards}]"
    print(f"{tag} boot torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)

    nodes_path = Path("/data/nodes.json")
    if not nodes_path.exists():
        sys.exit(f"{tag} FATAL: {nodes_path} missing")

    with open(nodes_path, encoding="utf-8") as f:
        nodes = json.load(f)
    n_total = len(nodes)

    # Equal-size slice for this shard
    per = (n_total + n_shards - 1) // n_shards
    lo = shard_index * per
    hi = min(lo + per, n_total)
    my_nodes = nodes[lo:hi]
    print(f"{tag} owns nodes[{lo}:{hi}] = {len(my_nodes)} chunks of {n_total}", flush=True)

    print(f"{tag} loading {MODEL_NAME}...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = MAX_LENGTH
    dim = model.get_sentence_embedding_dimension()
    print(f"{tag} model loaded in {time.time()-t0:.1f}s, dim={dim}", flush=True)

    texts = [f"{(n.get('title') or '')}\n\n{n.get('text','')}" for n in my_nodes]
    n_my = len(texts)
    n_batches = (n_my + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"{tag} {n_my} chunks, batch={BATCH_SIZE}, {n_batches} batches, max_len={MAX_LENGTH}", flush=True)

    out = np.empty((n_my, dim), dtype=np.float32)
    t_start = time.time()
    last_heartbeat = t_start
    log_every = max(1, n_batches // 40)

    for bi in range(n_batches):
        b_lo = bi * BATCH_SIZE
        b_hi = min(b_lo + BATCH_SIZE, n_my)
        vecs = model.encode(
            texts[b_lo:b_hi],
            batch_size=b_hi - b_lo,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out[b_lo:b_hi] = vecs.astype(np.float32, copy=False)

        now = time.time()
        if bi % log_every == 0 or bi == n_batches - 1 or now - last_heartbeat > 30:
            elapsed = now - t_start
            done = b_hi
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n_my - done) / rate if rate > 0 else 0.0
            print(
                f"{tag} [{bi+1:>4}/{n_batches}] {done:>6}/{n_my} ({100*done/n_my:5.1f}%) "
                f"| {rate:6.1f}/s | elapsed {elapsed/60:5.1f}m | eta {eta/60:5.1f}m",
                flush=True,
            )
            last_heartbeat = now

    total = time.time() - t_start
    print(f"{tag} DONE in {total/60:.1f}m  throughput={n_my/total:.1f} chunks/s", flush=True)

    vec_out = Path(f"/data/vectors_bgem3_shard{shard_index}.npy")
    id_out = Path(f"/data/id_map_bgem3_shard{shard_index}.json")
    np.save(vec_out, out)
    ids_for_shard = {i: n["id"] for i, n in enumerate(my_nodes)}  # local index within shard
    with open(id_out, "w", encoding="utf-8") as f:
        json.dump({"start": lo, "end": hi, "ids": [n["id"] for n in my_nodes]}, f, ensure_ascii=False)
    volume.commit()
    print(f"{tag} wrote {vec_out} ({out.nbytes/1e9:.2f} GB)", flush=True)
    return {"shard": shard_index, "n": n_my, "dim": dim, "seconds": total, "lo": lo, "hi": hi}


@app.function(volumes={"/data": volume}, timeout=600)
def upload_nodes(payload: bytes):
    out = Path("/data/nodes.json")
    out.write_bytes(payload)
    volume.commit()
    print(f"[upload] wrote {out} ({len(payload)/1e6:.1f} MB)", flush=True)
    return len(payload)


@app.function(volumes={"/data": volume}, timeout=600)
def merge_shards(n_shards: int) -> dict:
    """Concatenate shards into final vectors_bgem3.npy + id_map_bgem3.json."""
    import numpy as np
    pieces = []
    all_ids = []
    for i in range(n_shards):
        vec = np.load(f"/data/vectors_bgem3_shard{i}.npy")
        meta = json.load(open(f"/data/id_map_bgem3_shard{i}.json"))
        assert meta["start"] == sum(len(p) for p in pieces), \
            f"shard {i} starts at {meta['start']} but cumulative={sum(len(p) for p in pieces)}"
        pieces.append(vec)
        all_ids.extend(meta["ids"])
        print(f"[merge] shard {i}: {vec.shape}", flush=True)
    full = np.concatenate(pieces, axis=0)
    print(f"[merge] full shape: {full.shape}", flush=True)
    np.save("/data/vectors_bgem3.npy", full)
    id_map = {i: nid for i, nid in enumerate(all_ids)}
    with open("/data/id_map_bgem3.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)
    volume.commit()
    return {"n": full.shape[0], "dim": int(full.shape[1])}


@app.function(volumes={"/data": volume}, timeout=600)
def download_artifacts() -> dict:
    vec = Path("/data/vectors_bgem3.npy").read_bytes()
    idm = Path("/data/id_map_bgem3.json").read_bytes()
    return {"vectors": vec, "id_map": idm}


@app.local_entrypoint()
def main(shards: int = 4):
    local_nodes = Path("data/nodes.json")
    if not local_nodes.exists():
        sys.exit(f"FATAL: {local_nodes} missing locally")

    print(f"[local] uploading nodes.json ({local_nodes.stat().st_size/1e6:.1f} MB)...", flush=True)
    upload_nodes.remote(local_nodes.read_bytes())

    print(f"[local] launching {shards} parallel embed shards on {GPU}...", flush=True)
    args = [(i, shards) for i in range(shards)]
    t0 = time.time()
    results = list(embed_shard.starmap(args))
    print(f"[local] all shards done in {(time.time()-t0)/60:.1f}m: "
          f"{sum(r['n'] for r in results)} chunks total", flush=True)

    print(f"[local] merging shards on Modal...", flush=True)
    merged = merge_shards.remote(shards)
    print(f"[local] merged: {merged}", flush=True)

    print(f"[local] downloading artifacts...", flush=True)
    arts = download_artifacts.remote()
    Path("data/vectors_bgem3.npy").write_bytes(arts["vectors"])
    Path("data/id_map_bgem3.json").write_bytes(arts["id_map"])
    print(f"[local] wrote data/vectors_bgem3.npy ({len(arts['vectors'])/1e9:.2f} GB)", flush=True)
    print(f"[local] wrote data/id_map_bgem3.json ({len(arts['id_map'])/1e6:.1f} MB)", flush=True)
