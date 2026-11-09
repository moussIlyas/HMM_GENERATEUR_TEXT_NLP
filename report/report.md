---
title: "Lab 03: Chaotic Text Generator. One-page report"
---

## 1. Problem and solution

- Text generation samples new sentences from a probabilistic model, but a generated sentence has no reference answer, so quality has to be measured on hundreds of sentences.
- A first-order HMM only sees the previous tag, so after a NOUN it cannot tell a subject (a verb follows) from an object (a preposition or full stop follows).
- We build a POS-HMM (hidden tags, word emissions) estimated by counting, with temperature sampling and Viterbi tagging in log space.
- We add role-split states (S / O / P) so the model knows where it is in the sentence, and compare with a trigram baseline on the same data.
- A Streamlit test bench (`app.py`) generates batches, shows hidden tags, computes four metrics, tags typed sentences and exports CSV.

## 2. Metrics at T = 1.0 (500 sentences, seed 42)

| Model | Valid tag pattern | Plausible subject-verb | Novel sentences | Distinct-2 |
|---|---|---|---|---|
| HMM, plain states | 16.2% | 14.8% | 99.8% | 0.198 |
| HMM, role-split states | 100% | 60.6% | 99.4% | 0.148 |
| Trigram | 57.0% | 76.0% | 91.2% | 0.105 |

The plain HMM scores low because after a NOUN its next tag is VERB 0.39, PREP 0.22 and PUNCT 0.39 whatever the role of the noun, so it often ends a sentence after an ADJ NOUN or builds an invalid pattern. Role-split states give the model the memory of whether it is before the verb, after the verb or inside a prepositional phrase. This removes every invalid pattern (100%). The trigram remembers two previous words, so it reaches 57% on patterns, but it cannot enforce the full sentence shape.

## 3. Choosing a temperature

- **Writing toy: T = 1.5 (role-split HMM).** Structure stays at 100% valid and distinct-2 rises to 0.161, so the output is varied and still grammatical; surprises are welcome in a toy.
- **Autocomplete: T = 0.5 (role-split HMM).** Structure stays at 100% valid and predictions stay close to the most likely continuation; a low temperature avoids odd suggestions. The trigram should not be used here, because at T = 0.5 it copies training sentences.

## 4. Failure examples (T = 1.0, seed 42)

| Sentence | Failure type | Cause |
|---|---|---|
| "the quiet school ." (plain HMM) | **Structure**: tag pattern DET ADJ NOUN PUNCT has no verb, so it is not in the 16 training patterns | First-order model: after ADJ NOUN the next tag PUNCT is possible, so the sentence can end without a verb. |
| "the foxes lose by the school ." (role-split HMM, from `results/failures.txt`) | **Meaning**: valid pattern, but the pair foxes-lose never occurs in training | The states encode role, not word meaning, so any noun can be paired with any verb of the same role. |
| "the girls leave the park ." (trigram, T = 0.5) | **Memorisation**: copied from the training set | At low temperature the trigram counts are sharp, so it repeats a seen sentence. |

Good example (role-split HMM, T = 1.0, not in training): "the teachers eat the birds near the river ." It has a valid pattern and a plausible subject-verb pair, but it is only fluent, not meaningful: the model has no idea of what teachers or birds are.

## 5. What state splitting changed, and what it did not

State splitting changed the structure of the output: valid tag patterns rose from 16.2% to 100%, and plausible subject-verb pairs rose from 14.8% to 60.6%, because the verb is now chosen with the subject role in mind. It also made the hidden states more readable (DET_S, NOUN_O, PREP ...). It did not change the vocabulary, so word choices are still arbitrary and meaning is not modelled. Novelty stayed near 99%, while distinct-2 fell from 0.198 to 0.148 (fewer distinct word pairs, since the structure is now more constrained). Plausible subject-verb (60.6%) is still below the trigram (76.0%). Tagging accuracy on the synthetic corpus was 100% both before and after the split, because the grammar is unambiguous once the previous tag is known.

## 6. Synthetic versus real text

| Model | Accuracy | Known words | Unknown words | Whole sentence correct |
|---|---|---|---|---|
| Most-frequent tag (baseline) | 86.2% | 90.8% | 28.6% | 31.9% |
| HMM + Viterbi | 88.1% | 92.9% | 28.2% | 40.4% |
| *Synthetic corpus, HMM + Viterbi* | *100%* | *100%* | *n/a (no unseen words)* | *100%* |

The HMM gains only 1.9 points over the most-frequent-tag baseline on English EWT (17 tags, 16,654-word vocabulary, 10,359 distinct tag sequences for 12,544 sentences). Unknown words are 7.5% of test tokens and are tagged at only 28.2% accuracy; errors go to rare tags such as X and INTJ (PROPN tagged X 195 times, INTJ 137 times). The synthetic corpus has 6 tags, 61 words and only 16 tag sequences, so Viterbi is trivially right there. Real text has a large vocabulary, many unseen words and almost one tag sequence per sentence, so the synthetic results do not transfer. The valid-tag-pattern metric is also meaningless on real text, since it is defined relative to 16 synthetic sequences.
