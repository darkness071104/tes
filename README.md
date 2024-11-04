# Gender Classification Using InceptionV3

This project implements a gender classification model using the InceptionV3 architecture pre-trained on ImageNet. The model can classify images as either Male or Female based on the input data.

## Requirements

Before running the code, ensure that you have the following Python libraries installed:

- TensorFlow
- Pandas
- NumPy
- Matplotlib

You can install these dependencies using pip:

bash pip install tensorflow pandas numpy matplotlib


## Dataset

The model uses a dataset located at:

- **Training Directory:** `/path/to/your/training/dataset` (update this path)
- **Validation Directory:** `/path/to/your/validation/dataset` (update this path)

Make sure your datasets are organized in separate subdirectories for each class (e.g., `Male` and `Female`) within the `Training` and `Validation` directories.

## Usage

1. **Load the Dataset:**
   Update the `train_dir` and `val_dir` variables to point to your dataset directories.

2. **Image Preprocessing:**
   The program uses `ImageDataGenerator` to preprocess the images by rescaling pixel values to the range [0, 1].

3. **Model Creation:**
   The InceptionV3 model is created with an additional Global Average Pooling layer and a Dense layer for binary classification.

4. **Model Training:**
   Train the model using the `.fit()` method on the training data, with validation held against the validation set.

5. **Saving the Model:**
   Uncomment the saving line in the code to save the trained model as `inceptionv3_gender_classification.h5`.

6. **Load the Trained Model:**
   To load the model for inference, ensure you specify the correct path to the `.h5` file.

7. **Predict Gender:**
   - Update the path `img_path` with the location of the image you wish to classify.
   - Call the `predict_gender_with_image()` function with the model and the image path to see the prediction results.


## Example Output

When you run the prediction function, a plot will be displayed showing the input image along with the predicted gender and confidence level as follows:

Prediction: Male (85.50%)
