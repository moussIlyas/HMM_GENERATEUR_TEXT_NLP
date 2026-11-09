"""Lab 03 - Chaotic Text Generator test bench (Streamlit UI). Run:  streamlit run app.py"""
import csv
import io
import random
import re

import streamlit as st

from src.pipeline import HMM, NGramGenerator, coarse, generation_metrics, load_tagged, split_by_document

st.set_page_config(page_title="Chaotic Text Generator", layout="wide")
st.title("Chaotic Text Generator")
st.warning("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined relative "
           "to this corpus and do not measure the quality of real English.")

MODELS = {"Plain HMM": "hmm", "Role-split HMM": "hmm_split", "Trigram": "ngram"}


@st.cache_resource
def get_models():
    docs = load_tagged()
    train, _ = split_by_document(docs)
    return train, HMM(train), HMM(train, split_states=True), NGramGenerator([w for w, _ in train])


train, hmm_plain, hmm_split, ng = get_models()
train_patterns = {" ".join(t) for _, t in train}

st.header("Generator")
c1, c2, c3, c4 = st.columns(4)
model_name = c1.selectbox("Model", list(MODELS))
temperature = c2.slider("Temperature (chaos)", 0.3, 3.0, 1.0, 0.1)
seed = c3.number_input("Random seed", 0, 10_000, 42)
n_sentences = c4.slider("Number of sentences", 10, 500, 500, 10)

if st.button("Generate", type="primary"):
    kind = MODELS[model_name]
    rng = random.Random(int(seed))
    rows = []
    for _ in range(n_sentences):
        if kind == "ngram":
            words = ng.sample_sentence(rng, temperature)
            states = hmm_split.viterbi(words)
        else:
            model = hmm_split if kind == "hmm_split" else hmm_plain
            words, states = model.sample_with_states(rng, temperature)
        rows.append((words, states))
    metrics = generation_metrics([(w, coarse(s)) for w, s in rows], train, hmm_split)
    st.session_state["batch"] = {
        "model": model_name,
        "temperature": temperature,
        "seed": int(seed),
        "rows": [(" ".join(w), " ".join(s), " ".join(coarse(s)) in train_patterns) for w, s in rows],
        "metrics": metrics,
    }

batch = st.session_state.get("batch")
if batch:
    st.subheader(f"{batch['model']}, T = {batch['temperature']}, seed {batch['seed']}, {len(batch['rows'])} sentences")
    st.caption("For the trigram model the hidden tags are Viterbi tags from the role-split HMM.")
    st.dataframe(
        [{"#": i, "Sentence": s, "Hidden tags": t, "Valid pattern": "✔" if ok else "✘"}
         for i, (s, t, ok) in enumerate(batch["rows"], 1)],
        hide_index=True,
    )

    m = batch["metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valid tag pattern", f"{m['valid_tag_pattern']:.1%}",
              help="Share of sentences whose tag sequence is one of the training sequences")
    k2.metric("Plausible subject-verb", f"{m['plausible_subject_verb']:.1%}",
              help="Share of sentences whose first noun and first verb form a pair seen in training")
    k3.metric("Novelty", f"{m['novel_sentences']:.1%}", help="Share of sentences not copied from the training set")
    k4.metric("Distinct-2", f"{m['distinct_2']:.3f}", help="Distinct word bigrams / total word bigrams")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["sentence", "hidden_tags", "valid_tag_pattern"])
    for s, t, ok in batch["rows"]:
        writer.writerow([s, t, ok])
    st.download_button("Download CSV", buf.getvalue(),
                       file_name=f"generated_{batch['model'].lower().replace(' ', '_')}_T{batch['temperature']}"
                                 f"_seed{batch['seed']}.csv",
                       mime="text/csv")

st.header("Tagger (Viterbi)")
sentence = st.text_input("Type a sentence", "the cats sleep in the kitchen.")
tokens = re.findall(r"\w+|[^\w\s]", sentence.lower())
if tokens:
    tags = hmm_plain.viterbi(tokens)
    unknown = [w for w in tokens if w not in hmm_plain.vocab]
    cells = [w.replace("|", "\\|") + (" ⚠" if w in unknown else "") for w in tokens]
    st.markdown("| " + " | ".join(cells) + " |\n|" + "---|" * len(tokens) + "\n| "
                + " | ".join(t.replace("|", "\\|") for t in tags) + " |")
    if unknown:
        st.warning(f"Not in the training vocabulary: {', '.join(dict.fromkeys(unknown))}. "
                   f"Their tags are smoothed guesses (⚠).")
else:
    st.info("Type a sentence to tag it.")
