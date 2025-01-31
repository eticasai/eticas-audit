<p align="center">
  <img src="https://raw.githubusercontent.com/eticasai/eticas-audit/main/docs/eticas.png" width="150">
</p>

# Eticas: Bias & Audit Framework


An open-source Python library designed for developers to calculate fairness metrics and assess bias in machine learning models. This library provides a comprehensive set of tools to ensure transparency, accountability, and ethical AI development.

<p align="center">
    <a href="https://eticas.ai/itaca/" target="_blank">
        <img alt="Blog" src="https://img.shields.io/website?up_message=ETICAS AI&url=https://eticas.ai/case-study-category/knowledge-center/">
    </a>
        <a href="https://github.com/eticasai/eticas-audit/releases">
        <img alt="Github Notes" src="https://img.shields.io/github/v/release/eticasai/eticas-audit">
    </a>
    </a>
        <a href="https://pypi.org/project/eticas-audit/">
        <img alt="pipy" src="https://img.shields.io/pypi/v/eticas-audit">
    </a>
    <a href="https://github.com/eticasai/eticas-audit/actions/workflows/lint.yml">
        <img alt="CI" src="https://github.com/eticasai/eticas-audit/actions/workflows/lint.yml/badge.svg">
    </a>
        <a href="https://github.com/eticasai/eticas-audit/actions/workflows/unit-tests.yml">
        <img alt="CI" src="https://github.com/eticasai/eticas-audit/actions/workflows/unit-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/eticasai/eticas-audit/blob/master/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" />
    </a>


![Flow calculate bias](https://raw.githubusercontent.com/eticasai/eticas-audit/main/docs/metric_flow.png)

<p align="center">
  <a href="https://eticas.ai/itaca/" target="_blank"> Website</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#%EF%B8%8F-metrics">Metrics</a> •
  <a href="#example-notebooks">Example Notebooks</a> •
  <a href="#quickStart-bias-auditing">QuickStart Bias Auditing</a> •
  <a href="#explore-results">Explore Results</a> •
</p>
    
## Why Use This Library?

AI System can inherit biases from data or amplify them during decision-making processes. This library helps ensure transparency and accountability by providing actionable insights to improve fairness in AI systems.

## 🚀 Key Features

This framework is designed to audit AI systems comprehensively across all stages of their lifecycle. At its core, it focuses on comparing privileged and underprivileged groups, ensuring a fair evaluation of model behavior. 

With a wide range of metrics, this framework is a game-changer in bias monitoring. It offers a deep perspective on fairness, allowing for comprehensive reporting even without relying on true labels. The only restriction for measuring bias in production lies in performance metrics, as they are directly tied to true labels.

The stages considered are the following:
1. The dataset used to train the model.
2. The dataset used in production.
3. A dataset containing the system’s final decisions, which may include human intervention or another model.




- **Demographic Benchmarking Monitoring**: Perform in-depth analysis of population distribution.
- **Model Fairness Monitoring**: Ensure equality and detect equity issues in decision-making.
- **Features Distribution Evaluation**: Analyze correlations, causality, and variable importance.
- **Performance Analysis**: Metrics to assess model performance, accuracy, and recall.
- **Model Drift Monitoring**: Detect and measure changes in data distributions and model behavior over time.

## 🌟 <span style="background-color:yellow">ITACA: Monitoring & Auditing Platform</span> 🌟


🟡 Unlock the full potential of Eticas by upgrading to our subscription model! With **ITACA**, our powerful SaaS platform, you can monitor every stage of your model’s lifecycle seamlessly. Easily integrate ITACA into your workflows with our library and API—start optimizing your models today!

- **Audit Subscription** 🔎: Stay compliant with major regulations and laws on bias and fairness.

Learn more about our platform at [🔗 ITACA – Monitoring & Auditing Platform](https://eticas.ai/itaca/).  



| <img src="https://raw.githubusercontent.com/eticasai/eticas-audit/main/docs/itaca_dash.jpg" width="400" /> | <img src="https://raw.githubusercontent.com/eticasai/eticas-audit/main/docs/itaca_overview.jpg" width="400" /> 


**COMING SOON** 🎉  
- **Developer Subscription** 🛠️: Connect to ITACA to monitor your models.
 



## ⚖️ Metrics


| **Group**                  | **Metric**             | **Label needed?** | **Description**                                                                                                                                         |
|----------------------------|------------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **fairness**               | d_equality            | no                | Analyze whether the system’s disparities occur because the model does not treat all groups equally.                                                    |
| **fairness**               | d_equity              | no                | Analyze whether the system’s disparities arise because some groups have unique characteristics and may need a boost.                                   |
| **fairness**               | d_parity              | no                | Calculate the ratio of Selection Rates (Disparate Impact, DI). It represents the chance of success.                                                     |
| **fairness**               | d_statisticalparity   | no                | Calculate the difference in Selection Rates (Statistical Parity Difference, SPD). It measures the gap in success rates.                                 |
| **fairness**               | d_calibrated_false    | yes               | Evaluate the calibration for negative outcomes across groups.                                                                                           |
| **fairness**               | d_calibrated_true     | yes               | Evaluate the calibration for positive outcomes across groups.                                                                                           |
| **fairness**               | d_equalodds_false     | yes               | Check whether false outcomes are distributed equally among groups.                                                                                      |
| **fairness**               | d_equalodds_true      | yes               | Check whether true outcomes are distributed equally among groups.                                                                                       |
| **Demographic Benchmarking** | da_inconsistency      | no                | Calculate the percentage of samples that belong to an underprivileged group.                                                                            |
| **Demographic Benchmarking** | da_positive           | no                | Calculate the percentage of samples that receive a positive outcome and belong to an underprivileged group.                                            |
| **Features Distribution**  | da_informative        | no                | Determine if there is a proxy feature in the dataset, meaning some features act as a protected attribute.                                              |
| **Features Distribution**  | dxa_inconsistency     | no                | Check if the protected attributes are highly related to the output.                                                                                    |
| **Performance**            | accuracy              | yes               | Calculate the proportion of correct predictions among all predictions.                                                                                  |
| **Performance**            | F1                    | yes               | Compute the harmonic mean of precision and recall.                                                                                                      |
| **Performance**            | precision             | yes               | Compute the ratio of true positives to all predicted positives.                                                                                        |
| **Performance**            | recall                | yes               | Compute the ratio of true positives to all actual positives.                                                                                           |
| **Performance**            | poor_performance      | yes               | Calculate the accuracy against the representation of the largest class.                                                                                 |
| **Drift**                  | Drift Train-Operational | no              | Evaluate changes in data or model performance between training and operational phases.                                                                 |


## 📥 Installation



```bash
pip install eticas-audit
```

## Example Notebooks

| Notebook | Description |
|-|-|
| [Audit AI System](example.ipynb) | Check how use this library to audit AI System. |


## QuickStart Bias Auditing

### Execute Audit

#### Define Senstive Attributes

Use a JSON object to define the sensitive attributes. You need to specify the columns where the attribute names and the underprivileged or privileged groups are defined. This definition can include a list to accommodate more than one group.

Sensitive attributes can be simple, for example sex or race. They can also be complex—for instance, the intersection of sex and race.

```json
{
    "sensitive_attributes": {
        "sex": {
            "columns": [
                {
                    "name": "sex",
                    "underprivileged": [2]
                }
            ],
            "type": "simple"
        },
        "ethnicity": {
            "columns": [
                {
                    "name": "ethnicity",
                    "privileged": [1]
                }
            ],
            "type": "simple"
        },
        "age": {
            "columns": [
                {
                    "name": "age",
                    "privileged": [3, 4]
                }
            ],
            "type": "simple"
        },
        "sex_ethnicity": {
            "groups": ["sex", "ethnicity"],
            "type": "complex"
        }
    }
}
```

#### Create Model

Initialize the model object that will be the focus of the audit. As important inputs, you need to define the sensitive attributes and specify which input features you want to analyze. It's not necessary to include all features—only the most important or relevant ones.

```python

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s - %(message)s'
)

from eticas.model.ml_model import MLModel
model = MLModel(
    model_name="ML Testing Regression",
    description="A logistic regression model to illustrate audits",
    country="USA",
    state="CA",
    sensitive_attributes=sensitive_attributes,
    features=["feature_0", "feature_1", "feature_2"]
)

```

#### Audit Labeled

This is how to define the audit for a labeled dataset. In general the dataset used to train the dataset.The required inputs are:  
1. **dataset_path** – path to the data,  
2. **label_column** – represents the true label,  
3. **output_column** – contains the model’s output,  
4. **positive_output** – A list of outputs considered positive.

You can also upload a label or output column with scoring, ranking, or recommendation values (continuous values). If the regression ordering is ascending, the positive output is interpreted as 1; if it is descending, it is interpreted as 0.


```python
model.run_labeled_audit(dataset_path ='files/example_training_binary_2.csv', 
                        label_column = 'outcome', 
                        output_column = 'predicted_outcome',
                        positive_output = [1])

#json labeled results
model.labeled_results


```

#### Audit Production

This is how to define the audit for a production dataset. The required inputs are:  
1. **dataset_path** – path to the data,  
3. **output_column** – contains the model’s output,  
4. **positive_output** – A list of outputs considered positive.

You can also upload a label or output column with scoring, ranking, or recommendation values (continuous values). If the regression ordering is ascending, the positive output is interpreted as 1; if it is descending, it is interpreted as 0.

```python
model.run_production_audit(dataset_path ='files/example_operational_binary_2.csv',
                           output_column = 'predicted_outcome',
                           positive_output = [1])

#json production results
model.production_results
```

#### Audit Impacted

This is how to define the audit for a production dataset. The required inputs are:  
1. **dataset_path** – path to the data,  
3. **output_column** – contains the model’s output,  
4. **positive_output** – A list of outputs considered positive.

You can also upload a label or output column with scoring, ranking, or recommendation values (continuous values). If the regression ordering is ascending, the positive output is interpreted as 1; if it is descending, it is interpreted as 0.

```python

model.run_impacted_audit(dataset_path ='files/example_impact_binary_2.csv', 
                         output_column = 'recorded_outcome',
                         positive_output = [1])

#json impacted results
model.impacted_results
```

#### Audit Drift

```python
model.run_drift_audit(dataset_path_dev = 'files/example_training_binary_2.csv', 
                      output_column_dev = 'outcome',
                      positive_output_dev = [1],
                      dataset_path_prod = 'files/example_operational_binary_2.csv', 
                      output_column_prod = 'predicted_outcome',
                      positive_output_prod = [1])

#json drift results
model.drift_results
```

### Explore Results

The results can be exported in JSON or DataFrame format. Both options allow you to extract the information with or without normalization. Normalized values range from 0 to 100, where 0 represents a poor result and 100 represents a perfect value.

```python
audit_result = model.df_results(norm_values=True)
```

```python
audit_result = model.json_results(norm_values=True)
```

### Metrics WITHOUT TRUE LABEL

#### Fairness

```python

result = audit_result.xs(('fairness',), level=(0,))
result = result.reset_index()
result = result.pivot(
    index=['metric', 'attribute'],  
    columns='stage',             
    values='value'   
)

result

```


| Metric             | Attribute      | 01-labeled     | 02-production  | 03-impact      |
|--------------------|----------------|----------------|----------------|----------------|
| d_equality         | age            | 99.0  ▓▓▓▓▓▓   | 99.0  ▓▓▓▓▓▓   | 100.0 ▓▓▓▓▓▓   |
|                    | ethnicity      | 100.0 ▓▓▓▓▓▓   | 99.0  ▓▓▓▓▓▓   | 44.0  ▓▓▓      |
|                    | sex            | 100.0 ▓▓▓▓▓▓   | 71.0  ▓▓▓▓     | 47.0  ▓▓▓      |
|                    | sex_ethnicity  | 99.0  ▓▓▓▓▓▓   | 62.0  ▓▓▓▓     | 44.0  ▓▓▓      |
| d_equity           | age            | 100.0 ▓▓▓▓▓▓   | 99.0  ▓▓▓▓▓▓   | 100.0 ▓▓▓▓▓▓   |
|                    | ethnicity      | 100.0 ▓▓▓▓▓▓   | 70.0  ▓▓▓▓     | 44.0  ▓▓▓      |
|                    | sex            | 99.0  ▓▓▓▓▓▓   | 48.0  ▓▓▓      | 41.0  ▓▓▓      |
|                    | sex_ethnicity  | 100.0 ▓▓▓▓▓▓   | 57.0  ▓▓▓      | 44.0  ▓▓▓      |
| d_parity           | age            | 98.0  ▓▓▓▓▓▓   | 98.0  ▓▓▓▓▓▓   | 99.0  ▓▓▓▓▓▓   |
|                    | ethnicity      | 99.0  ▓▓▓▓▓▓   | 76.0  ▓▓▓▓     | 42.0  ▓▓▓      |
|                    | sex            | 99.0  ▓▓▓▓▓▓   | 64.0  ▓▓▓▓     | 40.0  ▓▓▓      |
|                    | sex_ethnicity  | 99.0  ▓▓▓▓▓▓   | 65.0  ▓▓▓▓     | 40.0  ▓▓▓      |
| d_statisticalparity| age            | 98.0  ▓▓▓▓▓▓   | 98.0  ▓▓▓▓▓▓   | 98.0  ▓▓▓▓▓▓   |
|                    | ethnicity      | 100.0 ▓▓▓▓▓▓   | 74.0  ▓▓▓▓     | 43.0  ▓▓▓      |
|                    | sex            | 100.0 ▓▓▓▓▓▓   | 57.0  ▓▓▓     | 42.0  ▓▓▓      |
|                    | sex_ethnicity  | 98.0  ▓▓▓▓▓▓   | 57.0  ▓▓▓     | 42.0  ▓▓▓      |


#### Benchmarking

```python

result = audit_result.xs(('benchmarking',), level=(0,))
result = result.reset_index()
result = result.pivot(
    index=['metric', 'attribute'],  
    columns='stage',             
    values='value'   
)

result

```


| Metric            | Attribute      | 01-labeled     | 02-production  | 03-impact      |
|-------------------|----------------|----------------|----------------|----------------|
| da_inconsistency  | age            | 44.8 ▓▓▓       | 44.5 ▓▓▓       | 45.0 ▓▓▓       |
|                   | ethnicity      | 40.0 ▓▓▓       | 20.0  ▓▓       | 10.0 ▓         |
|                   | sex            | 60.0 ▓▓▓▓      | 30.0  ▓▓       | 15.0  ▓        |
|                   | sex_ethnicity  | 25.0 ▓▓        | 15.0  ▓        | 10.0 ▓         |
| da_positive       | age            | 45.2 ▓▓▓       | 44.9 ▓▓▓       | 45.3  ▓▓▓      |
|                   | ethnicity      |  39.8  ▓▓▓     |  15.8 ▓        |  3.7 ▓         |
|                   | sex            |  59.8  ▓▓▓▓    |  19.7  ▓▓      | 5.7 ▓          |
|                   | sex_ethnicity  |  24.7  ▓▓      | 10.0  ▓        | 3.7  ▓         |

#### Distribution

```python

result = audit_result.xs(('distribution',), level=(0,))
result = result.reset_index()
result = result.pivot(
    index=['metric', 'attribute'],  
    columns='stage',             
    values='value'   
)

result

```


| Metric             | Attribute      | 01-labeled     | 02-production  | 03-impact      |
|--------------------|----------------|----------------|----------------|----------------|
| d_equality         | age            | 100.0 ▓▓▓▓▓▓   | 100.0 ▓▓▓▓▓▓   | 100.0 ▓▓▓▓▓▓   |
|                    | ethnicity      | 100.0 ▓▓▓▓▓▓   | 100.0 ▓▓▓▓▓▓   | 72.0  ▓▓▓▓     |
|                    | sex            | 99.0  ▓▓▓▓▓▓   | 90.0  ▓▓▓▓▓▓   | 60.0  ▓▓▓▓     |
|                    | sex_ethnicity  | 100.0 ▓▓▓▓▓▓   | 90.0  ▓▓▓▓▓▓   | 60.0  ▓▓▓▓     |
| d_equity           | age            | 99.0 ▓▓▓▓▓▓    | 98.0  ▓▓▓▓▓▓   | 98.0  ▓▓▓▓▓▓   |
|                    | ethnicity      | 99.0  ▓▓▓▓▓▓   | 89.0  ▓▓▓▓▓▓   | 77.0  ▓▓▓▓     |
|                    | sex            | 99.0  ▓▓▓▓▓▓   | 77.0  ▓▓▓▓▓▓   | 70.0  ▓▓▓▓     |
|                    | sex_ethnicity  | 98.0  ▓▓▓▓▓▓   | 86.0  ▓▓▓▓▓▓   | 77.0  ▓▓▓▓     |


#### Drift

```python

result = audit_result.xs(('drift',), level=(0,))
result = result.reset_index()
result = result.pivot(
    index=['metric', 'attribute'],  
    columns='stage',             
    values='value'   
)

result

```


| Metric             | Attribute      | 02-production  |
|--------------------|----------------|----------------|
| drift              | age            | 1.17 ▓         |
|                    | ethnicity      | 0.0  ▓         |
|                    | overall        | 0.87 ▓         |
|                    | sex            | 0.0  ▓         |
|                    | sex_ethnicity  | 1.62 ▓         |


### Metrics WITH TRUE LABEL

#### Fairness

```python

result = audit_result.xs(('fairness_label',), level=(0,))
result = result.reset_index()
result = result.pivot(
    index=['metric', 'attribute'],  
    columns='stage',             
    values='value'   
)

result

```


| Metric             | Attribute      | 01-labeled     |
|--------------------|----------------|----------------|
| d_calibrated_false | age            | 99.0  ▓▓▓▓▓▓   |
|                    | ethnicity      | 97.0  ▓▓▓▓▓▓   |
|                    | sex            | 98.0  ▓▓▓▓▓▓   |
|                    | sex_ethnicity  | 99.0  ▓▓▓▓▓▓   |
| d_calibrated_true  | age            | 96.0  ▓▓▓▓▓▓   |
|                    | ethnicity      | 98.0  ▓▓▓▓▓▓   |
|                    | sex            | 96.0  ▓▓▓▓▓▓   |
|                    | sex_ethnicity  | 95.0  ▓▓▓▓▓▓   |
| d_equalodds_false  | age            | 99.0  ▓▓▓▓▓▓   |
|                    | ethnicity      | 97.0  ▓▓▓▓▓▓   |
|                    | sex            | 98.0  ▓▓▓▓▓▓   |
|                    | sex_ethnicity  | 99.0  ▓▓▓▓▓▓   |
| d_equalodds_true   | age            | 96.0  ▓▓▓▓▓▓   |
|                    | ethnicity      | 98.0  ▓▓▓▓▓▓   |
|                    | sex            | 96.0  ▓▓▓▓▓▓   |
|                    | sex_ethnicity  | 95.0  ▓▓▓▓▓▓   |


#### Performance

```python

result = audit_result.xs(('performance',), level=(0,))
result = result.reset_index()
result = result.pivot(
    index=['metric', 'attribute'],  
    columns='stage',             
    values='value'   
)

result

```

| Metric             | Attribute      | 01-labeled     |
|--------------------|----------------|----------------|
| FN                 | age            | 1151           |
|                    | ethnicity      | 987            |
|                    | sex            | 1475           |
|                    | sex_ethnicity  | 619            |
| FP                 | age            | 1146           |
|                    | ethnicity      | 1027           |
|                    | sex            | 1565           |
|                    | sex_ethnicity  | 653            |
| TN                 | age            | 1060           |
|                    | ethnicity      | 973            |
|                    | sex            | 1428           |
|                    | sex_ethnicity  | 605            |
| TP                 | age            | 1121           |
|                    | ethnicity      | 1013           |
|                    | sex            | 1532           |
|                    | sex_ethnicity  | 623            |
| accuracy           | age            | 48.7  ▓▓       |
|                    | ethnicity      | 49.65 ▓▓       |
|                    | sex            | 49.33 ▓▓       |
|                    | sex_ethnicity  | 49.12 ▓▓       |
| f1                 | age            | 49.39 ▓▓       |
|                    | ethnicity      | 50.15 ▓▓       |
|                    | sex            | 50.2  ▓▓       |
|                    | sex_ethnicity  | 49.48 ▓▓       |
| poor_performance   | age            | 57.58 ▓▓       |
|                    | ethnicity      | 59.58 ▓▓       |
|                    | sex            | 59.05 ▓▓       |
|                    | sex_ethnicity  | 58.57 ▓▓       |
| precision          | age            | 49.45 ▓▓       |
|                    | ethnicity      | 49.66 ▓▓       |
|                    | sex            | 49.47 ▓▓       |
|                    | sex_ethnicity  | 48.82 ▓▓       |
| recall             | age            | 49.34 ▓▓       |
|                    | ethnicity      | 50.65 ▓▓       |
|                    | sex            | 50.95 ▓▓       |
|                    | sex_ethnicity  | 50.16 ▓▓       |

