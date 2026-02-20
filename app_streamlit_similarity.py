import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import imagehash
import streamlit as st

# CLIP deps
import torch
import open_clip


DATA_DIR = Path("margo_data")
CSV_PATH = DATA_DIR / "artworks.csv"
IMG_DIR = DATA_DIR / "images"

# pHash cache
EMB_PATH = DATA_DIR / "phash_16x16.npy"  # Larger hash

# CLIP index
CLIP_INDEX_PATH = DATA_DIR / "auc_clip_index.pkl"

# Thresholds / defaults
MIN_PHASH_SIMILARITY = 70.0  # Only show matches >=70/100
DEFAULT_TOP_K = 6


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df[df["local_image_path"].astype(str).str.len() > 0].copy()
    return df


# ----------------------------
# pHash pipeline (existing)
# ----------------------------
@st.cache_data
def compute_phashes(df):
    if EMB_PATH.exists():
        return np.load(EMB_PATH)

    hashes = []
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            img = Image.open(row["local_image_path"])
            phash = imagehash.phash(img, hash_size=16)
            hashes.append(np.array(phash.hash.flatten(), dtype=float))
        except Exception:
            hashes.append(np.zeros(256))

    embs = np.array(hashes)
    np.save(EMB_PATH, embs)
    return embs


def get_query_hash(uploaded_img):
    img = Image.open(uploaded_img).convert("RGB")
    phash = imagehash.phash(img, hash_size=16)
    return np.array(phash.hash.flatten(), dtype=float)


def find_matches_phash(query_hash, phash_matrix, top_k=DEFAULT_TOP_K):
    dists = np.sum(np.abs(phash_matrix - query_hash), axis=1)  # Hamming: 0-256
    scores = (256 - dists) / 256 * 100.0

    valid = scores >= MIN_PHASH_SIMILARITY
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return [], np.array([])

    sorted_valid = valid_idx[np.argsort(-scores[valid_idx])]
    top_idx = sorted_valid[:top_k]
    top_scores = scores[top_idx]
    return top_idx, top_scores


# ----------------------------
# CLIP pipeline (new)
# ----------------------------
@st.cache_data
def load_clip_index():
    if not CLIP_INDEX_PATH.exists():
        return None
    with open(CLIP_INDEX_PATH, "rb") as f:
        idx = pickle.load(f)
    return idx


@st.cache_resource
def load_clip_model(model_name: str):
    # Must match how the index was built (model_name + pretrained tag)
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, preprocess, device


def clip_embed_image(pil_img: Image.Image, model, preprocess, device) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    img_t = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)  # shape (1, D)


def find_matches_clip(query_vec: np.ndarray, idx: dict, top_k=DEFAULT_TOP_K):
    """
    idx keys (from your inspection):
      - filenames: list[str]
      - embeddings: np.ndarray shape (N, D)
    We compute cosine similarity: (N,D) dot (D,)
    """
    embs = idx["embeddings"].astype(np.float32)

    # Ensure normalized embeddings (safe even if already normalized)
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    q = query_vec.reshape(-1).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    sims = embs_norm @ q  # shape (N,)
    top_idx = np.argsort(-sims)[:top_k]
    top_scores = sims[top_idx]
    return top_idx.tolist(), top_scores.tolist()


def build_filename_to_row_map(df: pd.DataFrame) -> dict:
    """
    Map '0121.jpg' -> dataframe row index
    Uses local_image_path filenames.
    """
    out = {}
    for i, row in df.iterrows():
        p = str(row.get("local_image_path", ""))
        name = Path(p).name
        if name:
            out[name] = i
    return out


def main():
    st.set_page_config(page_title="Margo Veillon Search", layout="wide")
    st.title("Margo Veillon Similarity Search")
    st.markdown("Upload an image and retrieve the closest matches from the AUC collection.")

    df = load_data()
    if df.empty:
        st.error("No data found. Ensure margo_data/artworks.csv and images exist.")
        st.stop()

    mode = st.radio(
        "Search mode",
        ["CLIP (robust for photos/scans)", "pHash (best for exact duplicates)"],
        horizontal=True,
    )

    top_k = st.slider("Number of results", 3, 12, DEFAULT_TOP_K, 1)

    uploaded_file = st.file_uploader(
        "Upload image (JPG/PNG/TIFF)",
        type=["jpg", "jpeg", "png", "tiff"],
        help="For best results, crop close to the artwork.",
    )
    if not uploaded_file:
        st.info("Upload an image to start.")
        st.stop()

    col1, col2 = st.columns([1, 3])
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", width=280)

    # Common mapping for CLIP results -> df rows
    filename_to_row = build_filename_to_row_map(df)

    if mode.startswith("pHash"):
        phash_matrix = compute_phashes(df)
        with st.spinner("Computing pHash similarity..."):
            query_hash = get_query_hash(uploaded_file)
            match_idx, scores = find_matches_phash(query_hash, phash_matrix, top_k=top_k)

        if len(match_idx) == 0:
            st.warning("No strong pHash matches found. Try CLIP mode or crop closer to the artwork.")
            st.stop()

        st.subheader("Top Matches")
        cols = st.columns(min(3, len(match_idx)))
        for i, (idx_row, score) in enumerate(zip(match_idx, scores)):
            row = df.iloc[idx_row]
            col = cols[i % len(cols)]
            with col:
                try:
                    thumb = Image.open(row["local_image_path"])
                    st.image(thumb, use_column_width=True)
                except Exception:
                    st.error("Could not open match image.")

                st.markdown(f"**{str(row['title'])[:55]}**")
                st.caption(f"Score: {score:.0f}/100 | ID: {row['item_id']}")
                st.caption(f"Date: {row.get('date', '')}")
                st.caption(f"Collection: {row.get('collection', '')}")
                st.caption(f"Material: {row.get('material', '')}")
                st.caption(f"Technique: {row.get('technique', '')}")
                st.markdown(f"[AUC Source]({row.get('detail_url', '#')})")

    else:
        idx = load_clip_index()
        if idx is None:
            st.error("CLIP index not found. Build it first: margo_data/auc_clip_index.pkl")
            st.stop()

        model_name = idx.get("model_name", "ViT-B-16")
        model, preprocess, device = load_clip_model(model_name)

        with st.spinner("Computing CLIP similarity..."):
            pil = Image.open(uploaded_file)
            qvec = clip_embed_image(pil, model, preprocess, device)
            match_idx, sims = find_matches_clip(qvec, idx, top_k=top_k)

        st.subheader("Top Matches")

        cols = st.columns(min(3, len(match_idx)))
        for i, (j, sim) in enumerate(zip(match_idx, sims)):
            fname = idx["filenames"][j]
            df_row_idx = filename_to_row.get(Path(fname).name)

            col = cols[i % len(cols)]
            with col:
                if df_row_idx is None:
                    st.warning(f"No CSV row found for {fname}.")
                    continue

                row = df.loc[df_row_idx]

                try:
                    thumb = Image.open(row["local_image_path"])
                    st.image(thumb, use_column_width=True)
                except Exception:
                    st.error("Could not open match image.")

                st.markdown(f"**{str(row['title'])[:55]}**")
                st.caption(f"CLIP similarity: {sim*100:.1f}% | ID: {row['item_id']}")
                st.caption(f"Date: {row.get('date', '')}")
                st.caption(f"Collection: {row.get('collection', '')}")
                st.caption(f"Material: {row.get('material', '')}")
                st.caption(f"Technique: {row.get('technique', '')}")
                st.markdown(f"[AUC Source]({row.get('detail_url', '#')})")

    with st.expander("Debug"):
        st.write(f"Rows in CSV with images: {len(df)}")
        if mode.startswith("CLIP"):
            st.write(f"CLIP index items: {load_clip_index().get('num_items') if load_clip_index() else 'N/A'}")
            st.write(f"CLIP model_name: {load_clip_index().get('model_name') if load_clip_index() else 'N/A'}")


if __name__ == "__main__":
    main()
