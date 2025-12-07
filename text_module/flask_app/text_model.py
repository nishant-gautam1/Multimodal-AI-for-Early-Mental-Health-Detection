# app.py (fixed build_mapping to avoid ambiguous truth value)
import os
import pickle
from pathlib import Path
from flask import Flask, request, jsonify, render_template

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR.parent / "model"
HF_MODEL_DIR = MODELS_DIR / "saved_mental_status_bert"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
PRED_LOG = BASE_DIR / "predictions_log.csv"

if not PRED_LOG.exists():
    with open(PRED_LOG, "w", encoding="utf-8") as f:
        f.write("text,mapped_label,raw_label,index,score\n")

# load label encoder
label_encoder = None
try:
    if LABEL_ENCODER_PATH.exists():
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print("Loaded label encoder. classes_:", getattr(label_encoder, "classes_", None))
    else:
        print("label_encoder.pkl not found at", LABEL_ENCODER_PATH)
except Exception as e:
    print("Failed to load label encoder:", e)
    label_encoder = None

# load HF model & tokenizer
hf_tokenizer = None
hf_model = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if HF_MODEL_DIR.exists():
        hf_tokenizer = AutoTokenizer.from_pretrained(str(HF_MODEL_DIR))
        hf_model = AutoModelForSequenceClassification.from_pretrained(str(HF_MODEL_DIR))
        hf_model.eval()
        hf_model.to("cpu")
        print("Loaded HF model from", HF_MODEL_DIR)
    else:
        print("HF model folder not found at", HF_MODEL_DIR)
except Exception as e:
    print("Error loading HF model/tokenizer:", e)
    hf_tokenizer = None
    hf_model = None

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))


def _normalize_id2label(raw):
    """
    Ensure id2label is a dict mapping int->str.
    Accepts:
      - dict already
      - list/tuple/ndarray of labels where index is the label id
      - other types -> return None
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        new = {}
        for k, v in raw.items():
            try:
                ik = int(k)
            except Exception:
                try:
                    ik = int(str(k))
                except Exception:
                    continue
            new[ik] = v
        return new
    try:
        if isinstance(raw, (list, tuple)):
            return {i: raw[i] for i in range(len(raw))}
        import numpy as _np
        if isinstance(raw, _np.ndarray):
            return {int(i): str(raw[i]) for i in range(raw.shape[0])}
    except Exception:
        pass
    return None

def build_mapping():
    """
    Build mapping index -> encoder_label robustly.
    Returns a dict or None.
    """
    if hf_model is None:
        return None
    raw_id2label = getattr(hf_model.config, "id2label", None)
    id2label = _normalize_id2label(raw_id2label)
    try:
        num_labels = int(getattr(hf_model.config, "num_labels", None))
    except Exception:
        num_labels = None
    enc_classes = None
    if label_encoder is not None:
        try:
            enc_classes_raw = getattr(label_encoder, "classes_", None)
            if enc_classes_raw is not None:
                try:
                    enc_classes = list(enc_classes_raw)
                except Exception:
                    enc_classes = [str(x) for x in enc_classes_raw]
        except Exception:
            enc_classes = None
    # Strategy 1: name-based matching (case-insensitive)
    if id2label is not None and enc_classes is not None:
        id2lower = {i: str(l).strip().lower() for i, l in id2label.items()}
        enc_lower = [str(c).strip().lower() for c in enc_classes]

        mapping = {}
        matched = 0
        for i, name in id2lower.items():
            if name in enc_lower:
                mapped_label = enc_classes[enc_lower.index(name)]
                mapping[int(i)] = mapped_label
                matched += 1

        if matched == len(enc_classes) and len(mapping) == len(enc_classes):
            print("Mapping built by matching id2label -> encoder.classes_ (case-insensitive).")
            return mapping
        print(f"Name-based mapping matched {matched}/{len(enc_classes)} classes; skipping partial mapping.")
    # Strategy 2: if lengths match, build by index
    if enc_classes is not None and num_labels is not None and len(enc_classes) == num_labels:
        mapping = {i: enc_classes[i] for i in range(num_labels)}
        print("Mapping built by index alignment (encoder.classes_ length == model.num_labels).")
        return mapping
    # Strategy 3: use id2label directly if no encoder
    if id2label is not None and num_labels is not None and (enc_classes is None):
        # use id2label values as mapping strings
        mapping = {i: id2label.get(i) for i in range(num_labels)}
        print("Mapping built from model.id2label (no encoder present).")
        return mapping

    print("Could not build an automatic mapping between model indices and encoder labels.")
    return None

# Build mapping
index_to_encoder_label = build_mapping()

def map_index_to_label(idx, hf_label=None):
    if index_to_encoder_label is not None:
        return index_to_encoder_label.get(int(idx)), "encoder_map"
    if label_encoder is not None:
        try:
            mapped = label_encoder.inverse_transform([int(idx)])[0]
            return mapped, "label_encoder.inverse_transform"
        except Exception:
            pass
    id2label = _normalize_id2label(getattr(hf_model.config, "id2label", None)) if hf_model else None
    if id2label and int(idx) in id2label:
        return id2label[int(idx)], "id2label"
    return str(idx), "raw_index"

# ---------- Inference ----------
def infer_text_internal(text):
    if hf_model is None or hf_tokenizer is None:
        raise RuntimeError("Model/tokenizer not loaded")

    import torch
    inputs = hf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    device = next(hf_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hf_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        score = float(probs[idx])

    id2label = _normalize_id2label(getattr(hf_model.config, "id2label", None))
    raw_label = id2label[idx] if id2label and idx in id2label else None

    mapped_label, method = map_index_to_label(idx, raw_label)

    # log prediction
    try:
        safe_text = text.replace("\n", " ").replace("\r", " ").replace(",", " ")
        with open(PRED_LOG, "a", encoding="utf-8") as f:
            f.write(f"\"{safe_text}\",{mapped_label},{raw_label if raw_label else ''},{idx},{score}\n")
    except Exception:
        pass

    return {
        "mapped_label": mapped_label,
        "mapping_method": method,
        "raw_label_from_config": raw_label,
        "index": idx,
        "score": score,
        "probs": probs,
        "id2label": id2label
    }

# ---------- Flask endpoints ----------
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "<h3>Flask server — model loaded: {}</h3><p>Use /api/predict (POST JSON) or /predict (form)</p>".format(hf_model is not None)

@app.route("/predict", methods=["POST"])
def predict_form():
    text = request.form.get("text_input", "").strip()
    if not text:
        return render_template("index.html", input_text="", prediction="Please type some text.")
    try:
        result = infer_text_internal(text)
    except Exception as e:
        result = {"error": str(e)}
    return render_template("index.html", input_text=text, prediction=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    try:
        result = infer_text_internal(text)
        result_out = {
            "label": result["mapped_label"],
            "mapping_method": result["mapping_method"],
            "index": result["index"],
            "score": result["score"],
            "probs": result["probs"],
            "raw_label_from_config": result["raw_label_from_config"]
        }
        return jsonify(result_out)
    except Exception as e:
        return jsonify({"error": "inference failed", "detail": str(e)}), 500

@app.route("/api/predict_debug", methods=["POST"])
def api_predict_debug():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    text = request.get_json().get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    try:
        res = infer_text_internal(text)
        debug = {
            "text": text,
            "result": res,
            "label_encoder_classes": list(getattr(label_encoder, "classes_", [])) if label_encoder is not None else None,
            "model_id2label": _normalize_id2label(getattr(hf_model.config, "id2label", None))
        }
        return jsonify(debug)
    except Exception as e:
        return jsonify({"error": "debug inference failed", "detail": str(e)}), 500

@app.route("/debug_labels", methods=["GET"])
def debug_labels():
    info = {}
    info["model_id2label"] = _normalize_id2label(getattr(hf_model.config, "id2label", None)) if hf_model else None
    info["model_num_labels"] = getattr(hf_model.config, "num_labels", None) if hf_model else None
    info["label_encoder_classes"] = list(getattr(label_encoder, "classes_", [])) if label_encoder is not None else None
    info["mapping_built"] = index_to_encoder_label if index_to_encoder_label is not None else None
    return jsonify(info)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": hf_model is not None})

@app.route("/routes", methods=["GET"])
def routes():
    return jsonify([{"rule": r.rule, "methods": list(r.methods)} for r in app.url_map.iter_rules()])

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
