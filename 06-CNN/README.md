# Challenge-06

## Instructions

1. **Task**: *Pet identification* with Convolutional Neural Networks. Recognize which specific animal (12 classes, individual dogs and cats) is depicted in each test image, maximizing the **accuracy** on the test set. The suggested approach fine-tunes a **MobileNet V1** with TensorFlow 2.x / Keras on the raw RGB images and predicts the class of each test image independently of the others. The final classifier must be **one single neural network**: MobileNet V1 plus optional additional layers (no ensembles of any kind). Explore:
   - the layers added on top of the base network (number of layers and neurons per layer) and their hyperparameters;
   - the optimizer (any of those available in Keras) and its hyperparameters, possibly with a learning rate scheduler;
   - the minibatch size and the number of epochs;
   - pre-trained ImageNet weights vs randomly initialized weights;
   - a training scheme in which only some layers are trained at different moments (`base_model.trainable`), e.g. head first, then fine-tuning of the whole network;
   - **data augmentation** (scale, rotation, translation, brightness changes, etc.), following the [official TensorFlow guide](https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en#apply_augmentation_to_a_dataset), or applied directly on the NumPy images before creating the TF `Dataset`;
   - early stopping / saving the model of the epoch with the best validation accuracy.
2. **Suggestion**: Hold out a stratified validation set from the training images (e.g. with `train_test_split`, `stratify=label_y`) to tune the hyperparameters, and use the confusion matrix to spot the confused animals. Once the configuration is chosen, retrain the single network and apply the same preprocessing (resizing to 128x128) to the test set. The threshold for full marks is an accuracy on the test set **greater than or equal to 90%**.

## Rules & Constraints
- **Allowed Models**: **MobileNet V1 in its "small" version**, i.e. input resolution **128x128** and **width multiplier (alpha) 0.5**, with a custom final part (the layers added after the base network). The only model must be a **single neural network** composed of MobileNet V1 with optional additional layers.
- **Allowed**:
  - Free variation of the layers added on top of the base network and of their hyperparameters.
  - Tuning of the minibatch size, the number of epochs, the learning rate, and the choice of the optimizer (and its hyperparameters) among those available in [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers).
  - **One learning rate scheduler** among those available in [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules).
  - MobileNet V1 with either **ImageNet pre-trained weights** or **randomly initialized weights**.
  - Training schemes in which only some layers are trained at different moments (`base_model.trainable`).
  - **Data augmentation** of the training images (e.g. scale, rotation, translation, brightness).
  - **Preprocessing** of the images (e.g. rescaling, normalization).
- **Forbidden**:
  - **Modifying the architecture of the base network** (MobileNet V1), or using a different network or a different version of MobileNet (resolution other than 128, width multiplier other than 0.5).
  - **Ensembles of models of any kind** (e.g. averaging the outputs of several networks, of several training runs or of several checkpoints, or any other multiclassifier): the only model must be the single neural network made of MobileNet V1 with optional additional layers.
  - **External images** in the training set, i.e. expanding the dataset with images that do not come from the provided one (the test images are included).
- **Data Leakage Rule**: Fit any preprocessing strictly on the training set. Apply to the test set only the transformations learned/defined on the training set. Data augmentation must be applied to the training images only, never to the validation or test images.

## Datasets
Images of dogs and cats collected by students of this course. The same dataset supports a binary problem (dog vs cat) and a multiclass problem (single animal). **This challenge uses only the multiclass problem**: the binary problem is to be ignored.

- Train: image list `MulticlassTrainingSet.txt`, with the images in `DBs/CaniGatti_ML18_Es8_2020`. Class names (12 classes, one per animal) are in `DBs/CaniGatti_ML18_Es8_2020/MulticlassLabels.txt`.- Test: image list `MulticlassTestSet.txt`, **without labels**. It contains the same patterns as the test set of the binary problem, but the task is different (12 classes instead of 2).
Images must be resized to 128x128 before being fed to the network.

## Output File Requirements
- **Predictions**: `Es8Predictions.txt`
  - One line per test image: the predicted class index (integer, as given by the `argmax` of the output probabilities), **in the same order** as the test set.
  - No header, no index, no commas.
  - Can be generated with `np.savetxt('Es8Predictions.txt', predicted_y.astype(int), fmt='%i')`.
- **Submission ZIP**: `<filename>.zip` with the following structure:
  ```text
  <filename>.zip
  ├── Codice/
  │   └── <all code (notebooks and scripts) used for training and testing the model>
  └── Es8Predictions.txt
  ```
  **NB**: `Es8Predictions.txt` must be at the root of the zip, **not** inside a folder. The `Codice` folder must contain everything needed to replicate the result (a copy of the notebook and of the Python scripts used).