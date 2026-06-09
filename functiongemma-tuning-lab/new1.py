def _render_dataset_tab(engine_state):
    with gr.TabItem("1. Preparing Dataset"):
        gr.Markdown("### 🛠️ Tool Schema & Data Import")
        gr.Markdown("**Important Limitation:** This configuration will fail if the defined tools require **different parameter structures**.<br>The framework cannot currently handle a mix of tools with distinct signatures. For example, the following combination will not work:")
        gr.Markdown("* `sum(int a, int b)`\n* `query(string q)`")
        gr.Markdown("Ensure that all tools within this specific schema definition share a consistent parameter format.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Step 1: Define Functions**<br>Edit the JSON schema below to define the tools the model should learn.")
                tools_editor = gr.Code(language="json", label="Tool Definitions (JSON Schema)", lines=15)
                update_tools_btn = gr.Button("💾 Update Tool Schema")
                tools_status = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("**Step 2: Upload Data (Optional)**<br>To train on your own data, upload a CSV file to replace the [default dataset](https://huggingface.co/datasets/bebechien/SimpleToolCalling).")
                gr.Markdown("**Example CSV Row:** No header required.<br>Format: `[User Prompt, Tool Name, Tool Args JSON]`\n```csv\n\"What is the weather in London?\", \"get_weather\", \"{\"\"location\"\": \"\"London, UK\"\"}\"\n```")
                import_file = gr.File(label="Upload Dataset (.csv)", file_types=[".csv"], height=100)
                import_status = gr.Markdown("")
    
    # ========== 新增推理演示面板 ==========
    gr.Markdown("---")
    gr.Markdown("### 🚀 Inference Demo 推理演示")
    with gr.Row():
        with gr.Column(scale=2):
            # 演示Prompt下拉选择
            demo_prompt_select = gr.Dropdown(
                label="Select Demo Prompt 选择演示Prompt",
                choices=[],  # 动态加载
                interactive=True
            )
            # 自定义Prompt输入
            custom_prompt = gr.Textbox(
                label="Custom Prompt 自定义Prompt",
                lines=3,
                placeholder="Enter your prompt here...",
                interactive=True
            )
            # 推理按钮
            run_inference_btn = gr.Button("▶️ Run Inference 执行推理", variant="primary")
        
        with gr.Column(scale=3):
            # 推理输出（双输出：简洁+详细，这里合并为带格式的输出）
            inference_output = gr.Textbox(
                label="Inference Output 推理输出",
                lines=10,
                interactive=False,
                autoscroll=True
            )
    
    # Return controls needed for wiring
    return {
        "tools_editor": tools_editor,
        "update_tools_btn": update_tools_btn,
        "tools_status": tools_status,
        "import_file": import_file,
        "import_status": import_status,
        # 新增推理面板控件
        "demo_prompt_select": demo_prompt_select,
        "custom_prompt": custom_prompt,
        "run_inference_btn": run_inference_btn,
        "inference_output": inference_output
    }
