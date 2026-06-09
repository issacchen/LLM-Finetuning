---
title: FunctionGemma Tuning Lab
short_description: How to fine-tune FunctionGemma for tool calling
emoji: 📊
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
 - manage-repos
license: apache-2.0
---

# FunctionGemma Tuning Lab

**FunctionGemma Tuning Lab** is a user-friendly, Gradio-based interface demo designed to help you fine-tune [FunctionGemma](https://huggingface.co/google/functiongemma-270m-it) model to understand and utilize your specific custom tools and functions.

Whether you are building an agent to query internal knowledge bases, control smart home devices, or interact with proprietary APIs, this lab streamlines the process of teaching the model your specific function schemas.

## 
 Features

-   **Interactive Tool Definition:** Define your function schemas (JSON) directly in the UI.
-   **Custom Dataset Import:** Upload your own training data via CSV.
-   **One-Click Fine-Tuning:** Configure hyperparameters (Epochs, Learning Rate) and start training with a single click.
-   **Real-Time Monitoring:** Watch training logs and loss curves update in real-time.
-   **Automatic Evaluation:** The system automatically evaluates the model's performance before and after training to show improvement.
-   **Export Artifacts:** Download your fine-tuned model weights ready for deployment.

## 
 Installation

1.  **Clone the repository:**
    ```bash
    hf download google/functiongemma-tuning-lab --repo-type=space --local-dir=functiongemma-tuning-lab
    cd functiongemma-tuning-lab
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 
 Usage

1.  **Set up Environment Variables (Optional):**
    If you need to access gated models on Hugging Face, set your token:
    ```bash
    export HF_TOKEN=your_huggingface_token
    ```

2.  **Run the Application:**
    ```bash
    python app.py
    ```

3.  **Access the UI:**
    Open your browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

## 
 Data Format

To train on your own data, upload a CSV file with the following columns (no header required, or header ignored if present):

1.  **User Prompt:** The natural language query from the user.
2.  **Tool Name:** The name of the function that should be called.
3.  **Tool Arguments:** A JSON string representing the arguments for the function.

**Example CSV Row:**
```csv
"What is the weather in London?", "get_weather", "{""location"": ""London, UK""}"
```

## 
 Configuration

Core settings can be modified in `config.py`:
-   `MODEL_NAME`: Base model to fine-tune (default: `google/functiongemma-270m-it`).
-   `DEFAULT_DATASET`: Hugging Face dataset to use if no custom CSV is uploaded.
-   `ARTIFACTS_DIR`: Directory where training outputs are saved.

