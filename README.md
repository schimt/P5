# P5 Prediction of Traffic Flow using GCN 

This repository contains the implementation and analysis of the project **"Prediction of Traffic Flow"**. The study focuses on predicting traffic flow at 15-minute intervals within Aalborg Kommune, utilizing Graph Neural Networks (GNNs) and time series data to address the global challenge of traffic congestion.

---

## Project Details

**Theme:** Graph Data Analysis  
**Period:** Fall Semester 2024  
**Group:** cs-23-dvml-5-02  
**Institution:** Department of Computer Science, Aalborg University  

**Contributors:**  
- Gonde Leon Winkelmann  
- Laila Johnsen  
- Niklas Kjærgaard Andersen  
- Nikolai Schmidt Madsen  

**Supervisor:** Raffaele Pojer  

---

## Objectives

The project aims to:
1. Explore different graph representations to improve traffic flow prediction accuracy.
2. Evaluate the influence of lookback intervals and prediction horizons on model performance.
3. Analyze the effects of dataset scale on graph generation and prediction accuracy.

---

## Dataset

- **Source:** Danish traffic department (Vejdirektoratet).  
- **Coverage:** 485 traffic sensor points in Aalborg Kommune, focusing on the E45 highway.  
- **Timeframe:** January 2023 - August 2024.  
- **Resolution:** 15-minute aggregated intervals.  

Data preprocessing involved cleaning, handling missing values, and filtering to maintain continuity, resulting in datasets with 32 functional sensors and approximately 1.1 million individual traffic flow values.

---

## Methods and Implementation

### 1. Graph Construction
Various graph representations were created:
- **Fully Connected Graph**
- **Random Graph** (Erdős–Rényi model)
- **Distance-Based Graph** (Euclidean distances)
- **Correlation-Based Graphs** (Pearson and Cosine Similarities)
- **Delaunay Triangulated Graph**
- **Adaptive Threshold Graph** (Pearson correlation with thresholding)

Graph properties such as Total Number of Edges (TNE), Average Degree (AD), Density, Diameter, and Average Clustering Coefficient (ACC) were evaluated.

### 2. Graph Neural Networks (GNN)
The primary model architecture consisted of:
- Four GCN layers with ReLU activations.
- A dense output layer for predictions.

Additionally, a Graph Autoencoder (GAE) was implemented to encode and decode graph features, utilizing dropout for regularization.

### 3. Time Series Forecasting
Historical data was utilized to predict future traffic flow. Models were trained on lookback windows and evaluated for prediction horizons, exploring the trade-offs between computational efficiency and prediction accuracy.

---

## Results

Key findings:
1. **Graph Representation:** Graphs with geometrically or geographically informed connections (e.g., Delaunay and Distance-based) outperformed random or fully connected graphs.
2. **Lookback and Prediction Horizon:** Increasing the lookback window beyond 40 time steps (10 hours) showed diminishing returns in model performance.
3. **Best Model:** The Delaunay Unweighted graph provided the most consistent results across metrics (MAPE, VAPE, MSE).

---

## Evaluation Metrics

- **Mean Absolute Percentage Error (MAPE)**
- **Variance Absolute Percentage Error (VAPE)**
- **Mean Squared Error (MSE)**

Evaluation was conducted on datasets of varying scales (4-day and 1-year periods), highlighting the robustness of selected graph representations.

---

## Future Work

Future improvements may include:
- Integration of additional features (e.g., speed, anomalies).
- Analysis of nationwide traffic data.
- Implementation of advanced graph augmentation techniques.
- Optimization of hyperparameters for tailored performance.

---

## Conclusion

The study concludes that informed graph representations enhance the predictive capabilities of GNNs for traffic flow. While the results provide valuable insights, further research with more extensive data and advanced methodologies is recommended to refine the models.

---

## Repository Structure

- `data/`: Contains raw and processed datasets.
- `graphs/`: Scripts for graph construction and evaluation.
- `models/`: Implementation of GNN and GAE architectures.
- `results/`: Evaluation results and visualizations.
- `notebooks/`: Jupyter notebooks for exploratory analysis.

---

For further details, please refer to the project report.
