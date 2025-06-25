# One-Class Classification for Anomaly Detection

This repository contains the code and results for the BLG 454E course project on one-class classification. The project investigates the application of Deep Support Vector Data Description (Deep SVDD) for visual anomaly detection.

## About The Project

In many real-world scenarios, we only have access to data for what is considered "normal". This project tackles the problem of anomaly detection under this constraint. A Deep SVDD model was trained exclusively on a single "normal" class from the CIFAR-10 dataset and tasked with identifying all other classes as anomalies.

*   **Method:** Deep Support Vector Data Description (Deep SVDD)
*   **Dataset:** CIFAR-10
*   **Normal Class:** `automobile`
*   **Anomalous Classes:** All 9 other classes in CIFAR-10

## Getting Started

Follow these instructions to set up the project locally and run the experiment.

### Prerequisites

*   Python 3.8+
*   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required packages:**
    (First, ensure you have a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment).
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiment

To run the full training and evaluation process for both Deep SVDD and the One-Class SVM baseline, execute the main script:

```bash
python main.py
```

The script will:
1.  Automatically download the CIFAR-10 dataset (if not already present).
2.  Train the Deep SVDD model for 20 epochs.
3.  Evaluate the Deep SVDD model and save visualization results.
4.  Train and evaluate the One-Class SVM baseline and save its visualization.
5.  Print the final AUC scores to the console.

## Results

The models were evaluated using the Area Under the ROC Curve (AUC) metric.

### Quantitative Results

| Method                   | AUC Score |
| ------------------------ | --------- |
| Deep SVDD                | 0.6171    |
| One-Class SVM (Baseline) | 0.6172    |

### Qualitative Results

#### Training Loss
The training loss for the Deep SVDD model decreased consistently over 20 epochs, indicating that the model was successfully learning a compact representation.

![Deep SVDD Training Loss](loss_plot.png)

#### Anomaly Detection Visualization
The visualization shows examples of correct and incorrect classifications by the Deep SVDD model. The model struggles with visually similar anomalies (like a truck) and atypical normal instances.

![Deep SVDD Visualization](results/visualization_svdd.png)

## Conclusion

The project successfully demonstrated the implementation of a Deep SVDD model for one-class classification.

*   **Key Finding 1:** The model achieved a moderate AUC of ~0.62, showing it can distinguish anomalies better than chance.
*   **Key Finding 2:** The advanced Deep SVDD model did not outperform the simpler One-Class SVM baseline, suggesting that the quality of the pre-trained features was the most dominant factor in this experiment.
*   **Challenge:** Visual similarity between the normal class (`automobile`) and some anomalous classes (`truck`) proved to be the main limitation.

## Project File Structure
```
.
├── .gitignore
├── main.py                 # Main script for training and evaluation
├── plot_loss.py            # Script to generate the loss plot
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── loss_plot.png           # Output image of training loss
├── Final_Report.pdf        # Final written report
└── results/
    ├── visualization_ocsvm.png
    └── visualization_svdd.png
```
