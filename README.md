# Network DMD Prediction 
Dynamic Mode Decomposition for PredictingGraph Data

## Introduction

**Network DMD Prediction** is a project that leverages the **Dynamic Mode Decomposition with Control (DMDc)** algorithm and **Dynamic Mode Decomposition (DMD)** to analyze and predict graph data. This project also utilizes **node2vec**, a node embedding technique, to transform graph structures into vector spaces. Subsequently, DMDc is applied to model and forecast the dynamic changes of the graph over time. The embedded vector is used to control input of DMDc.

### Key Features

- **DMD Implementation**: Models dynamic systems to predict future states
- **DMDc Implementation**: Models dynamic systems with control inputs to predict future states.
- **Node Embedding**: Converts graph nodes into high-dimensional vectors using node2vec.
- **Data Visualization**: Visualizes model performance and prediction results for analysis.

## Installation

Follow the steps below to set up the project in your local environment.

### 1. Clone the Repository

```bash
git clone https://github.com/Dexoculus/Network-DMD-Prediction.git
cd Network-DMD-Prediction
```

### 2. intall Required Packages
```bash
pip install -r requirements.txt
```

## License
This project is provided under the MIT License.
