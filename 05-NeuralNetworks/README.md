# Challenge-05

## Instructions

1. **Task**: *Digit recognition* with neural networks. Recognize which digit (0-9, spoken in Italian) is pronounced in each of the 620 test tracks, maximizing the **accuracy** on the test set. The suggested approach trains a **Multilayer Perceptron (MLP)** with TensorFlow 2.x / Keras on the pre-extracted features and predicts the class of each test track independently of the others. Explore:
   - the architecture of the network (number of layers and neurons per layer);
   - the activation functions and the weight initialization (`kernel_initializer`);
   - the optimizer and its hyperparameters, possibly with a learning rate scheduler;
   - regularization (L1/L2 via `kernel_regularizer`, `Dropout`) and early stopping;
   - the choice of a subset of the features, or of derived features (e.g. MFCC derivatives with `librosa`);
   - the use of the MFCCs without temporal collapse (the tracks have a variable number of frames, so a way to process inputs of different sizes is needed);
   - a multiclassifier obtained by combining several neural networks.
2. **Suggestion**: The validation set must simulate the test conditions, i.e. unseen tracks recorded in different conditions. Split **by session**, e.g. train on session 1 and validate on session 4 (and vice versa), or use a cross-validation. Once the configuration is chosen, retrain the network on all training tracks (sessions 1 and 4) and apply the same preprocessing to the test set.

## Rules & Constraints
- **Allowed Models**: MLP networks made of `Dense` layers and optionally `Dropout` layers.
- **Allowed**:
  - Free variation of the architecture, activation functions, optimizer, learning rate scheduler, regularization and training hyperparameters.
  - **Preprocessing** of the data (e.g. rescaling, standardization).
  - Using only a **subset** of the features.
  - **Deriving new features** from existing ones (*augmentation* with derived features is allowed), e.g. MFCC derivatives computed with `librosa`.
  - Using the **MFCCs without temporal collapse** (field `mfccs`).
  - A **multiclassifier**, only by combining several neural networks.
- **Forbidden**:
  - **External data** in the training set, including the data of the test set.
  - Exploiting the number of test patterns or their distribution among the classes: every track must be **classified independently** of all the others.
  - Transformer models.
  - Classifiers other than MLP networks (e.g. SVM, Random Forest) alone or inside a multiclassifier.
- **Data Leakage Rule**: Fit any preprocessing strictly on the training set using `.fit_transform()`. Use `.transform()` on the test set.

## Datasets
Tracks recorded by 31 people, each repeating the ten digits in 4 sessions: (1) quiet room, (2) quiet room right after waking up, (3) noisy environment (e.g. outdoor traffic), (4) background TV or radio.

- Train: `<DBs/AudioDigits_ML26_27/audio_ml2627_train.pkl>`, sessions 1 and 4 (620 tracks = 2 sessions × 31 speakers × 10 digits). Pickled dictionary with aligned lists:
  - `features`: arrays of shape (173,), compact features of each track, split as MFCC `[0:20)`, Chroma `[20:32)`, mean of the Mel spectrogram `[32:160)`, Spectral Contrast `[160:167)`, Tonnetz `[167:173)`;
  - `mfccs`: arrays of shape (T, 20), the MFCCs of each track, where T is the number of frames (79-745);
  - `digits`: digit pronounced in each track (0-9), the label of this challenge;
  - `session_ids`: session of each track (1 or 4);
  - `person_ids`: speaker of each track (0-30), not needed for this challenge.
- Test: `<DBs/AudioDigits_ML26_27/audio_ml2627_test_no_labels.pkl>`, sessions 2 and 3 (620 tracks), with only the `features` and `mfccs` keys.

## Output File Requirements
- **Predictions**: `Es7Predictions.txt`
  - Exactly 620 lines: the predicted digit (0-9) of each test track, **in the same order** as the test set.
  - No header, no index, no commas.
- **Submission ZIP**: `<filename>.zip` with the following structure:
  ```text
  <filename>.zip
  ├── Codice/
  │   └── <all code (notebooks and scripts) used for training and testing the model>
  └── Es7Predictions.txt
  ```
  **NB**: `Es7Predictions.txt` must be at the root of the zip, **not** inside a folder.
