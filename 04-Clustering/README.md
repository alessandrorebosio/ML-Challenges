# Challenge-04

## Instructions

1. **Task**: *Speaker recognition* with clustering. Recognize which of the 31 speakers recorded each of the 620 test tracks, maximizing the **accuracy** on the test set. The suggested approach trains one **Gaussian Mixture Model (GMM)** per speaker on the MFCC frames of all its training tracks, and assigns each test track to the speaker whose GMM gives the highest **mean log-likelihood** (`gmm.score(X)`, not `gmm.predict(X)`, which returns the component of a single GMM). Explore:
   - preprocessing of the MFCCs (e.g. rescaling, standardization, deltas with `librosa.feature.delta(...)`);
   - hyperparameters of the GMMs (number of components $s$, covariance type, etc.), possibly different for each speaker;
   - other clustering techniques available in scikit-learn;
   - cleaning of anomalous data.
2. **Target**: Accuracy $\ge 65\%$ on the test set (threshold to obtain 2 points in the ranking).
3. **Suggestion**: The validation set must simulate the test conditions, i.e. unseen tracks recorded in different conditions. Split **by track, not by frame** (frames of the same track overlap and are almost identical), ideally **by session**: e.g. train on session 1 and validate on session 4. Once the configuration is chosen, retrain the GMMs on all training tracks (sessions 1 and 4) and apply the same preprocessing to the test set.

## Rules & Constraints
- **Allowed Models**: Any clustering technique available in scikit-learn, including Gaussian Mixture Models.
- **Allowed**:
  - Free variation of the clustering hyperparameters, also different for each speaker model.
  - **Preprocessing** of the data (e.g. rescaling, standardization).
  - **Reducing** the dataset by removing anomalous data (*cleaning*).
  - **Deriving new features** from existing ones (e.g. deltas with `librosa.feature.delta(...)`).
- **Forbidden**:
  - Classifiers of any kind: neither scikit-learn models nor neural networks or transformers.
  - Using the `features` and `digits` fields of the dataset: only the MFCCs may be used.
  - Extending the dataset (*augmentation*) by adding new patterns or creating derived ones from existing patterns.
- **Data Leakage Rule**: Fit any preprocessing strictly on the training set using `.fit_transform()`. Use `.transform()` on the test set.

## Datasets
Tracks recorded by 31 people, each repeating the ten digits in 4 sessions: (1) quiet room, (2) quiet room right after waking up, (3) noisy environment (e.g. outdoor traffic), (4) background TV or radio.

- Train: `<DBs/AudioDigits_ML26_27/audio_ml2627_train.pkl>`, sessions 1 and 4 (620 tracks = 2 sessions × 31 speakers × 10 digits, 20 tracks per speaker). Pickled dictionary with aligned lists:
  - `mfccs`: arrays of shape (T, 20), the MFCCs of each track, where T is the number of frames (79-745);
  - `person_ids`: speaker of each track (0-30);
  - `session_ids`: session of each track (1 or 4);
  - `features`, `digits`: not allowed for this challenge.
- Test: `<DBs/AudioDigits_ML26_27/audio_ml2627_test_no_labels.pkl>`, sessions 2 and 3 (620 tracks), with only the `features` and `mfccs` keys.

## Output File Requirements
- **Predictions**: `Es6Predictions.txt`
  - Exactly 620 lines: the predicted speaker id of each test track, **in the same order** as the test set.
  - No header, no index, no commas.
- **Submission ZIP**: `<filename>.zip` with the following structure:
  ```text
  <filename>.zip
  ├── Codice/
  │   └── <all code (notebooks and scripts) used for training and testing the model>
  └── Es6Predictions.txt
  ```
  **NB**: `Es6Predictions.txt` must be at the root of the zip, **not** inside a folder.
