# explanations.py

import shap
import lime
import lime.lime_tabular
import numpy as np

def generate_shap_explanation(model, data):
    """Generate SHAP explanation for model predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    return shap_values

def generate_lime_explanation(model, data, instance):
    """Generate LIME explanation for a single prediction."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(data),
        feature_names=data.columns,
        class_names=['Not Churn', 'Churn'],
        mode='classification'
    )
    explanation = explainer.explain_instance(instance, model.predict_proba)
    return explanation
