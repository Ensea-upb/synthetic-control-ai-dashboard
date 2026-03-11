# Synthetic Control AI Dashboard

Interactive data science dashboard for causal inference analysis using the **Synthetic Control method**.

This project provides a framework for evaluating the impact of public policies using observational data. The application constructs synthetic counterfactuals to estimate treatment effects, performs robustness diagnostics, and allows users to explore results through an interactive dashboard.

The system also integrates an **AI assistant** based on a multimodal transformer model to help users interpret results and generate analytical insights directly from the interface.

---

## Features

- Synthetic Control estimation for policy impact evaluation  
- Robustness diagnostics including placebo tests  
- Interactive visualizations of treated vs synthetic trajectories  
- Visualization of control unit weights  
- Integrated AI assistant for analytical interpretation  

---

## Technologies

- Python  
- Streamlit  
- pandas  
- matplotlib  
- HuggingFace Transformers  
- OpenVINO  

---

## Architecture

The application follows a modular architecture separating data processing, estimation methods, visualization, and AI integration.

Data → Processing → Synthetic Control → Visualization → Dashboard → AI Assistant


---

## Installation

Clone the repository:
git clone https://github.com/Ensea-upb/synthetic-control-ai-dashboard

cd synthetic-control-ai-dashboard

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app/0_Acceuil.py

## Local Models

The application relies on a pretrained AI model that is **not included in the repository** because of its large size.

To use the AI features, you need to manually download the model.

### Steps

1. Create a folder named `local_models` at the root of the project:

```bash
mkdir local_models
Download the pretrained model from the following link:

https://huggingface.co/OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov

Place the downloaded model inside the local_models folder.

Expected structure

sc_app/
│
├── local_models/
│   └── Qwen2.5-VL-7B-Instruct-int4-ov/
│
├── sc_core/
├── app_ui/
├── pages/
├── IA_integration/
└── requirements.txt
```
---

## AI Assistant

The dashboard integrates a multimodal transformer model **Qwen2-VL-7B** using HuggingFace Transformers.

The assistant helps users:

- interpret graphical results  
- explain causal estimation outputs  
- generate analytical insights  

Inference is optimized using **OpenVINO** to improve performance during model execution.

---

## Scientific Reference

The implementation of the Synthetic Control method is based on:

Abadie, A., Diamond, A., & Hainmueller, J. (2010)  
*“Synthetic Control Methods for Comparative Case Studies”*

Article link:  
https://hdl.handle.net/1721.1/144417

---

## Project Documentation

The repository also includes the project specification used to implement this application.
