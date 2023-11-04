# Data-Pipeline-Project
This project implements a Data Pipeline for analyzing loan data, including an Ingestion phase and a Transformation phase. The script (`data_pipeline.py`) processes loan data, handles missing values, transforms variables, and generates insights.
# Project Structure
- **data_pipeline.py**: Main script for the Data Pipeline.
   - Divided code into Ingestion and Transformation phases.
   - Handle exceptions for missing values, incorrect variable types, and out-of-range numerical values.
   - Transformatiom the dependent variable with the ln function.
# Guidelines and Considerations
Ensure you have the necessary dependencies installed (Pandas, tqdm, Matplotlib, scikit-learn).
The Data Pipeline follows the guidelines outlined in the project instructions:
1. **Prepare a Database:** A continuous dependent variable (loan_amount) and 10 independent variables (continuous, categorical, binary).
2. **Variable Types and Ranges:** For each variable, define the type (binary, categorical, continuous) and specify ranges and possible values.
3. **Textual to Numerical Values:** Replace textual values with numerical values.

# Alerts
The script provides alerts about the percentage of rows deleted from the total data.

## Additional Data Analysis
In addition to data pipeline, work was done in Excel and presentation of the insights in a PPT presentation in accordance with additional instructions for the project:
1. **Linear Relationships:** Check the linear relationship between each independent variable and the dependent variable.
2. **Relationship Between Variables:** Check the relationship between each independent variable and other variables.
3. **Pivot Table and Graphs:**
   - Produce a Pivot Table for each independent variable with the dependent variable.
   - Create suitable graphs according to variable types.    
4. **Generate Insights:** Generate initial insights/hypotheses from the data.
