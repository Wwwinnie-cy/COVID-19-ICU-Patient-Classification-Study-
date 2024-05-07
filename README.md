Purpose: Analyzed complex datasets from NIH Common Fund's National Metabolomics Data Repository (NMDR) to identify biomarkers and patterns associated with COVID-19 patient admissions to ICU.
Details:
Applied the point biserial correlation coefficient to extract 83 significant features and matched metabolites to identify key factors related to severe COVID-19 cases. 
Used t-SNE for dimensionality reduction analysis to reveal the distribution patterns of metabolite features across different ICU categories.
Employed Support Vector Machine (SVM), Random Forest (RF), Neural Networks (NN), XGBoost, and stacked models for classification tasks. Optimization was carried out using SMOTE and grid search, with model performance evaluated using five-fold cross-validation. 
The RF model achieved an accuracy of 0.71 in predicting ICU admissions.
