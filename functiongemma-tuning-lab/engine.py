import threading
import torch
import time
import json
import queue
import uuid
import matplotlib.pyplot as plt
from functools import partial
from typing import Generator, Optional, List, Dict, Any, Tuple
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from huggingface_hub import HfApi, model_info, metadata_update

from config import AppConfig
from tools import DEFAULT_TOOLS,DEFAULT_SYSTEM_MSG
from utils import (
    authenticate_hf, 
    load_model_and_tokenizer, 
    create_conversation_format, 
    parse_csv_dataset,
    zip_directory
)

class AbortCallback(TrainerCallback):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.stop_event.is_set():
            control.should_training_stop = True

class LogStreamingCallback(TrainerCallback):
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue
        
    def _get_string(self, value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        metrics_map = {
            "loss": "Loss",
            "eval_loss": "Eval Loss",
            "learning_rate": "LR",
            "epoch": "Epoch"
        }
        log_parts = [f"📝 [Step {state.global_step}]"]
        
        for key, label in metrics_map.items():
            if key in logs:
                val = logs[key]
                if isinstance(val, (float, int)):
                    val_str = f"{val:.4f}" if val > 1e-4 else f"{val:.2e}"
                else:
                    val_str = str(val)
            
                log_parts.append(f"{label}: {val_str}")
        
        log_payload = logs.copy()
        log_payload['step'] = state.global_step
        
        self.log_queue.put((" | ".join(log_parts), log_payload))

class FunctionGemmaEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        
        self.session_id = str(uuid.uuid4())[:8]
        self.output_dir = self.config.ARTIFACTS_DIR.joinpath(f"session_{self.session_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.loaded_model_name = None 
        self.imported_dataset = []
        self.stop_event = threading.Event()
        self.current_tools = DEFAULT_TOOLS
        self.has_model_tuned = False

        authenticate_hf(self.config.HF_TOKEN)
        try:
            self.refresh_model()
        except Exception as e:
            print(f"Initial load warning: {e}")

    # --- Tool Schema Methods ---
    def get_tools_json(self) -> str:
        return json.dumps(self.current_tools, indent=2)

    def update_tools(self, json_str: str) -> str:
        try:
            new_tools = json.loads(json_str)
            if not isinstance(new_tools, list):
                return "Error: Schema must be a list of tool definitions."
            self.current_tools = new_tools
            return "✅ Tool Schema Updated successfully."
        except json.JSONDecodeError as e:
            return f"❌ JSON Error: {e}"
        except Exception as e:
            return f"❌ Error: {e}"

    # --- Model & Data Management ---
    
    def _load_model_weights(self):
        print(f"[{self.session_id}] Loading model: {self.config.MODEL_NAME}...")
        self.model, self.tokenizer = load_model_and_tokenizer(self.config.MODEL_NAME)
        self.loaded_model_name = self.config.MODEL_NAME

    def refresh_model(self) -> str:
        self.has_model_tuned = False
        try:
            self._load_model_weights()
            return f"Model loaded: {self.loaded_model_name}\nData cleared.\nReady (Session {self.session_id})."
        except Exception as e:
            self.model = None
            self.tokenizer = None
            self.loaded_model_name = None
            return f"CRITICAL ERROR: Model failed to load. {e}"

    def load_csv(self, file_path: str) -> str:
        try:
            new_data = parse_csv_dataset(file_path)
            if not new_data:
                return "Error: File empty or format invalid."
            self.imported_dataset = new_data
            return f"Successfully imported {len(new_data)} samples."
        except Exception as e:
            return f"Import failed: {e}"

    def trigger_stop(self):
        self.stop_event.set()

    def _ensure_model_consistency(self) -> Generator[str, None, bool]:
        """Checks if the requested model matches the loaded one. Reloads if necessary."""
        if self.config.MODEL_NAME != self.loaded_model_name:
            yield f"🔄 Model changed. Switching from '{self.loaded_model_name}' to '{self.config.MODEL_NAME}'...\n"
            try:
                self._load_model_weights()
                yield "✅ Model reloaded successfully.\n"
                return True
            except Exception as e:
                yield f"❌ Failed to load model '{self.config.MODEL_NAME}': {e}\n"
                return False
        if self.model is None:
             yield "❌ Error: No model loaded.\n"
             return False
        return True

    # --- Evaluation Pipeline ---
    
    def run_evaluation(self, test_size: float, shuffle_data: bool) -> Generator[str, None, None]:
        self.stop_event.clear()
        output_buffer = ""
        
        try:
            # 1. Check Model
            gen = self._ensure_model_consistency()
            try:
                while True:
                    msg = next(gen)
                    output_buffer += msg
                    yield output_buffer
            except StopIteration as e:
                if not e.value: return # Failed to load
                
            # 2. Prepare Data
            output_buffer += f"⏳ Preparing Dataset for Eval (Test Split: {test_size})...\n"
            yield output_buffer

            dataset, log = self._prepare_dataset()
            output_buffer += log
            yield output_buffer
                
            if not dataset:
                output_buffer += "❌ Dataset creation failed.\n"
                yield output_buffer
                return

            if len(dataset) > 1:
                dataset = dataset.train_test_split(test_size=test_size, shuffle=shuffle_data)
            else:
                dataset = {"train": dataset, "test": dataset}
                
            # 3. Run Inference
            output_buffer += "\n📊 Evaluating Model Success Rate on Test Split...\n"
            yield output_buffer

            for update in self._evaluate_model(dataset["test"]):
                yield f"{output_buffer}{update}"
                if self.stop_event.is_set():
                    yield f"{output_buffer}{update}\n\n🛑 Evaluation interrupted by user."
                    break
        finally:
            self.stop_event.set() # Ensure loop breaks if generator cancelled

    # --- Training Pipeline ---

    def run_training_pipeline(self, epochs: int, learning_rate: float, test_size: float, shuffle_data: bool) -> Generator[Tuple[str, Any], None, None]:
        self.stop_event.clear()
        output_buffer = ""
        last_plot = None

        try:
            # 1. Check Model
            gen = self._ensure_model_consistency()
            try:
                while True:
                    msg = next(gen)
                    output_buffer += f"{msg}"
                    yield output_buffer, None
            except StopIteration as e:
                if not e.value: return

            output_buffer += f"⏳ Preparing Dataset (Test Split: {test_size}, Shuffle: {shuffle_data})...\n"
            yield output_buffer, None

            dataset, log = self._prepare_dataset()
            if not dataset:
                yield "Dataset creation failed.", None
                return

            output_buffer += log
            yield output_buffer, None
                
            if len(dataset) > 1:
                dataset = dataset.train_test_split(test_size=test_size, shuffle=shuffle_data)
            else:
                dataset = {"train": dataset, "test": dataset}

            # --- Training (Threaded) ---
            output_buffer += f"\n🚀 Starting Fine-tuning (Epochs: {epochs}, LR: {learning_rate})...\n"
            yield output_buffer, None
            
            log_queue = queue.Queue()
            training_error = None
            running_history = [] 
            
            def train_wrapper():
                nonlocal training_error
                try:
                    self._execute_trainer(dataset, log_queue, epochs, learning_rate)
                except Exception as e:
                    training_error = e
                    
            train_thread = threading.Thread(target=train_wrapper)
            train_thread.start()
            
            while train_thread.is_alive():
                while not log_queue.empty():
                    payload = log_queue.get()
                    if isinstance(payload, tuple):
                        msg, log_data = payload
                        output_buffer += f"{msg}\n"
                        running_history.append(log_data)
                        try:
                            last_plot = self._generate_loss_plot(running_history)
                            yield output_buffer, last_plot
                        except Exception:
                            yield output_buffer, last_plot
                    else:
                        output_buffer += f"{payload}\n"
                        yield output_buffer, last_plot
                
                if self.stop_event.is_set():
                    yield f"{output_buffer}🛑 Stop signal sent. Waiting for trainer to wrap up...\n", last_plot
                
                time.sleep(0.1)
            
            train_thread.join()
            
            self.has_model_tuned = True
            
            while not log_queue.empty():
                payload = log_queue.get()
                if isinstance(payload, tuple):
                    msg, log_data = payload
                    output_buffer += f"{msg}\n"
                    running_history.append(log_data)
                    last_plot = self._generate_loss_plot(running_history)
                else:
                    output_buffer += f"{payload}\n"
                yield output_buffer, last_plot
                    
            if training_error:
                output_buffer += f"❌ Error during training: {training_error}\n"
                yield output_buffer, last_plot
                return

            if self.stop_event.is_set(): 
                output_buffer += "🛑 Training manually stopped.\n"
                yield output_buffer, last_plot
                return
            
            output_buffer += "✅ Training finished.\n"
            yield output_buffer, last_plot
            
        finally:
            self.stop_event.set() # Ensure background thread stops if generator cancelled

    def _prepare_dataset(self):
        formatting_fn = partial(create_conversation_format, tools_list=self.current_tools)

        if not self.imported_dataset:
            ds = load_dataset(self.config.DEFAULT_DATASET, split="train").map(formatting_fn)
            log = f" `-> using default dataset (size:{len(ds)})\n"
        else:
            dataset_as_dicts = [{
                "user_content": row[0], "tool_name": row[1], "tool_arguments": row[2]}
                for row in self.imported_dataset
            ]
            ds = Dataset.from_list(dataset_as_dicts).map(formatting_fn)
            log = f" `-> using custom dataset (size:{len(ds)})\n"
        return ds, log

    def _execute_trainer(self, dataset, log_queue: queue.Queue, epochs: int, learning_rate: float) -> List[Dict]:
        torch_dtype = self.model.dtype
        args = SFTConfig(
            output_dir=str(self.output_dir),
            max_length=512,
            packing=False,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="epoch",
            learning_rate=learning_rate,
            fp16=(torch_dtype == torch.float16),
            bf16=(torch_dtype == torch.bfloat16),
            report_to="none",
            dataset_kwargs={"add_special_tokens": False, "append_concat_token": True}
        )

        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            processing_class=self.tokenizer,
            callbacks=[
                AbortCallback(self.stop_event),
                LogStreamingCallback(log_queue)
            ]
        )
        trainer.train()
        trainer.save_model()
        return trainer.state.log_history
        
    def _generate_loss_plot(self, history: list):
        if not history: return None
        plt.close('all')
        
        train_steps = [x['step'] for x in history if 'loss' in x]
        train_loss = [x['loss'] for x in history if 'loss' in x]
        eval_steps = [x['step'] for x in history if 'eval_loss' in x]
        eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

        fig, ax = plt.subplots(figsize=(10, 5))
        if train_steps:
            ax.plot(train_steps, train_loss, label='Training Loss', linestyle='-', marker=None)
        if eval_steps:
            ax.plot(eval_steps, eval_loss, label='Validation Loss', linestyle='--', marker='o')

        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        return fig

    def _evaluate_model(self, test_dataset) -> Generator[str, None, None]:
        results = []
        success_count = 0
        for idx, item in enumerate(test_dataset):
            messages = item["messages"][:2]
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages, tools=self.current_tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
                )
                device = self.model.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = self.model.generate(
                    **inputs, 
                    pad_token_id=self.tokenizer.eos_token_id, 
                    max_new_tokens=128
                )
                output = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                log_entry = f"{idx+1}. Prompt: {messages[1]['content']}\n   Output: {output[:100]}..."
                expected_tool = item['messages'][2]['tool_calls'][0]['function']['name']
                if expected_tool in output:
                    log_entry += "\n   -> ✅ Correct Tool"
                    success_count += 1
                else:
                    log_entry += f"\n   -> ❌ Wrong Tool (Expected: {expected_tool})"
                results.append(log_entry)
                yield "\n".join(results) + f"\n\nRunning Success Rate: {success_count}/{idx+1}"
            except Exception as e:
                yield f"Error during inference: {e}"

    def get_zip_path(self) -> Optional[str]:
        if not self.output_dir.exists(): return None
        base_name = str(self.config.ARTIFACTS_DIR.joinpath(f"functiongemma_finetuned_{self.session_id}"))
        return zip_directory(str(self.output_dir), base_name)

    def upload_model_to_hub(self, repo_name: str, oauth_token: str) -> str:
        """Uploads the trained model to Hugging Face Hub."""
        if not self.output_dir.exists() or not any(self.output_dir.iterdir()):
            return "❌ No trained model found in current session. Run training first."
        
        try:
            api = HfApi(token=oauth_token)

            # Get the authenticated user's username
            user_info = api.whoami()
            username = user_info['name']
        
            # Construct the full repo ID
            repo_id = f"{username}/{repo_name}"
            print(f"Preparing to upload to: {repo_id}")

            # Create the repo (safe if it already exists)
            api.create_repo(repo_id=repo_id, exist_ok=True)
            
            # Upload
            print(f"Uploading to {repo_id}...")
            repo_url = api.upload_folder(
                folder_path=str(self.output_dir),
                repo_id=repo_id,
                repo_type="model"
            )

            info = model_info(
                repo_id=repo_id,
                token=oauth_token
            )
            tags = ["functiongemma", "functiongemma-tuning-lab"]
            if info.card_data:
                tags = info.card_data.tags
                tags.append("functiongemma-tuning-lab")

            metadata_update(repo_id, {"tags": tags}, overwrite=True, token=oauth_token)

            return f"✅ Success! Model uploaded to: {repo_url}"
        except Exception as e:
            return f"❌ Upload failed: {str(e)}"


    # 在 engine.py 的 FunctionGemmaEngine 类中新增以下方法
    def run_inference(self, prompt: str) -> str:
        print("✅ 当前加载的工具:", self.current_tools)  # <-- 调试用
        import torch
        from transformers import GenerationConfig

        # ====================== 固定格式：必须传入 tools！======================
        messages = [
            {"role": "developer", "content": "You are a helpful assistant that calls tools."},
            {"role": "user", "content": prompt}
        ]

        # ====================== 核心：必须带 tools ！！！======================
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.current_tools,  # 必须传！不传就不会调用工具！
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 生成配置：强制输出结构化结果
        gen_cfg = GenerationConfig(
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)

        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):],
            skip_special_tokens=True
        )
        return response

    def get_demo_prompts(self) -> List[str]:
        """获取内置的演示Prompt列表"""
        return [
            "What is the weather in New York?",
            "Calculate 100 * 5 + 200",
            "Search knowledge base for project timeline",
            "Find the latest AI news on Google"
        ]
