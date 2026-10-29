# Pure-Acoustic Conformer-CTC Phoneme Recognizer

## Project Overview
This repository contains an end-to-end, Pure-Acoustic Automatic Speech Recognition (ASR) pipeline built entirely from scratch in PyTorch. The model is designed to transcribe raw audio directly into CMU ARPAbet phoneme sequences without the reliance on external Language Models (LMs). A core focus of this project is a mathematically verified **Domain Adaptation strategy** that significantly improves phonetic decoding accuracy on non-native English accents.

## Technical Architecture
* **Backbone:** 12-Layer Conformer (4 attention heads, 1024-dim Feedforward Network, depthwise convolution kernel of 31).
* **Modeling Paradigm:** Connectionist Temporal Classification (CTC) sequence-to-sequence modeling.
* **Vocabulary:** 41 discrete classes (39 standard CMU ARPAbet phonemes, 1 `` token, and 1 CTC `` token).
* **Feature Extraction:** 80-dimensional Log-Mel Spectrograms operating at 16 kHz with a 10 ms frame step. FP16 energy clamping (`min=1e-5`) is utilized to prevent gradient underflow during mixed-precision training.
* **Regularization:** SpecAugment (dynamic time and frequency masking) to mitigate overfitting on smaller, non-native datasets.

## Datasets & Ingestion Pipeline
The pipeline utilizes an on-the-fly Grapheme-to-Phoneme (G2P) conversion strategy using `g2p_en` to standardize phonetic targets across three distinct corpora, bypassing misaligned or missing dataset columns:
* **LibriSpeech (`train.100`):** Native English baseline data for foundational acoustic pre-training.
* **SpeechOcean762:** Multi-accented expansion data to introduce phonetic variance.
* **L2 Arctic:** The designated target domain of non-native speech, dynamically partitioned into strict 80% Train, 10% Validation, and 10% Unseen Test splits.

## Curriculum Learning Strategy
The training pipeline is split into two distinct phases to ensure broad generalization before specializing in the target domain:
* **Stage 1 (Acoustic Pre-Training):** 20,000 steps on LibriSpeech and SO762 using an AdamW optimizer and OneCycleLR scheduler to build a foundational understanding of English acoustics.
* **Stage 2 (Target Domain Adaptation):** 5,000 steps strictly on the L2 Arctic train partition. The learning rate is reduced by 10x (to `5e-5`) to perform delicate boundary adjustments on non-native accents without catastrophic forgetting of Stage 1 knowledge.

## Results & Ablation Study
The domain adaptation hypothesis was rigorously tested by evaluating unmasked acoustic features (bypassing SpecAugment during evaluation) on the validation and unseen test sets. 

* **Variant B (Baseline - Stage 1):** ~42% Phoneme Error Rate (PER)
* **Variant C (Domain Adapted - Stage 2):** ~32% Phoneme Error Rate (PER)

**Key Insight:** The 10% absolute reduction in PER was primarily driven by a sharp decrease in Deletion Errors. Fine-tuning allowed the model to confidently map ambiguous, non-native acoustic boundaries to actual phonemes rather than defaulting to the safe CTC `` token.

## Repository Structure
The pipeline is contained within a cleanly structured Google Colab notebook, divided into 8 core engineering phases:
1. **Environment Setup & Authentication:** Drive mounting and Hugging Face OAuth.
2. **Vocabulary & Data Ingestion:** 41-class deterministic mapping and stream partitioning.
3. **Model Architecture & Regularization:** Conformer initialization and SpecAugment.
4. **Curriculum Learning Stage 1:** 20k-step baseline mixed-precision training.
5. **Decoding & Quantitative Evaluation:** CTC greedy decoding and `jiwer` error analysis.
6. **Target Domain Adaptation (Stage 2):** 5k-step fine-tuning loop.
7. **Visualizations & Ablation Analysis:** Loss trajectory curves, ablation bar charts, and Spectrogram-to-Logit alignment graphs.
8. **Real-Time Inference:** Standalone audio processing and interactive UI.

## Interactive Inference
The project includes a standalone inference engine wrapped in a **Gradio v4** web interface. This allows for live microphone capture directly within the browser. The pipeline handles on-the-fly stereo-to-mono downmixing, 16 kHz resampling, feature extraction, and real-time CTC decoding for arbitrary `.wav` files or live voice recordings.

## Known Limitations
As a pure-acoustic, LM-free system, the model is highly sensitive to background noise and compression artifacts (e.g., standard web-audio compression). In live microphone testing, the model successfully identifies sharp consonant boundaries but exhibits aggressive deletion of continuous vowel sounds. Future iterations will require integrating an N-gram Language Model (e.g., KenLM) during the decoding phase to recover dropped vowels via linguistic context.
