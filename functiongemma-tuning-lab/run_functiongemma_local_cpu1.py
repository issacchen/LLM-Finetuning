# ======================
# 带 FULL DEBUG 版本
# 输出：变量 + 权重 + FFN + logits → debug_log.txt
# ======================
import torch
import json
import re
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer

# ======================
# DEBUG 日志工具（自动截断400字符）
# ======================
DEBUG_LOG = "debug_log.txt"

def log(msg, obj=None, max_len=400):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
        if obj is not None:
            s = str(obj)
            if len(s) > max_len:
                s = s[:max_len] + " ... [truncated]"
            f.write(s + "\n")
        f.write("-" * 80 + "\n")

# 清空旧日志
open(DEBUG_LOG, "w").close()
log("=== DEBUG START ===")

# ======================
# 配置
# ======================
max_seq_length = 512
model_name = "./functiongemma-270m-it"
log("CONFIG", {"max_seq_length": max_seq_length, "model_name": model_name})

# ======================
# 加载模型 & 分词器
# ======================
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
    output_hidden_states=True,  # 必须开，才能拿 FFN / 中间层
)

log("Tokenizer loaded", str(tokenizer))
log("Model loaded", str(model))

# ======================
# LoRA
# ======================
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
)
model = get_peft_model(model, lora_config)
log("LoRA config", lora_config)
model.print_trainable_parameters()

# ======================
# 工具定义
# ======================
def get_weather(city: str):
    return f"{city} 今日晴，气温 22~28℃"

def calculate(a: float, b: float, op: str):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b if b != 0 else 0

def date_calc(target_date: str, offset_days: int):
    from datetime import datetime, timedelta
    base = datetime.strptime(target_date, "%Y-%m-%d")
    res = base + timedelta(days=offset_days)
    return res.strftime("%Y-%m-%d")

FUNCTION_MAPPING = {
    "get_weather": get_weather,
    "calculate": calculate,
    "date_calc": date_calc
}

TOOLS = [
    {
        "name": "get_weather",
        "description": "获取城市天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "计算器",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "float"},
                "b": {"type": "float"},
                "op": {"type": "string"}
            },
            "required": ["a", "b", "op"]
        }
    },
    {
        "name": "date_calc",
        "description": "日期计算",
        "parameters": {
            "type": "object",
            "properties": {
                "target_date": {"type": "string"},
                "offset_days": {"type": "int"}
            },
            "required": ["target_date", "offset_days"]
        }
    }
]

log("TOOLS", TOOLS)

# ======================
# 训练数据
# ======================
train_data = [
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "从2025-01-01往后推15天是哪天？"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-01\",\"offset_days\":15}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-01-10往前倒退8天是什么日期"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-10\",\"offset_days\":-8}<end_function_call>"}]},
]

log("train_data count", len(train_data))
dataset = Dataset.from_list(train_data)

def format_template(examples):
    text = tokenizer.apply_chat_template(
        examples["messages"],
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_template)
log("Formatted dataset sample", dataset[0]["text"])

# ======================
# 训练
# ======================
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    warmup_steps=1,
    max_steps=5,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    logging_steps=1,
    optim="adamw_torch",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
)

log("Training args", training_args)
trainer.train()
log("=== TRAINING DONE ===")

# ======================
# 推理（带 DEBUG：权重 + FFN + logits）
# ======================
model.eval()
log("=== INFERENCE START ===")

messages = [
    {"role": "system", "content": "使用工具回答问题"},
    {"role": "user", "content": "2025-01-01加10天是哪天？"}
]
log("Input messages", messages)

inputs = tokenizer.apply_chat_template(
    messages,
    tools=TOOLS,
    add_generation_prompt=True,
    return_tensors="pt"
)
log("Model inputs shape", inputs.shape)
log("Input tokens", inputs)

# ======================
# 🔥 核心 DEBUG：权重 + FFN + logits
# ======================
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

    # 1. Logits
    logits = outputs.logits
    log("Logits shape", logits.shape)
    log("Logits sample (first 10 tokens)", logits[0, :10, :5])

    # 2. FFN 中间结果（最后一层 hidden state = FFN 输出）
    hidden_states = outputs.hidden_states
    last_hidden = hidden_states[-1]
    log("Last hidden state (FFN output)", last_hidden.shape)
    log("FFN hidden sample", last_hidden[0, :5, :10])

    # 3. 模型权重（关键层）
    for name, param in model.named_parameters():
        if "q_proj" in name or "up_proj" in name or "down_proj" in name:
            log(f"Weight: {name}", param.shape)
            log(f"Weight value sample: {name}", param[:5, :5])
            break

# 继续生成
with torch.no_grad():
    generate_outputs = model.generate(**inputs, max_new_tokens=64)

text = tokenizer.decode(generate_outputs[0], skip_special_tokens=True)
log("Final model output", text)
print("\n模型输出：\n", text)

# ======================
# 工具解析
# ======================
def extract_tools(text):
    pattern = r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    out = []
    for name, argstr in matches:
        try:
            args = json.loads("{" + argstr + "}")
            out.append({"name": name, "args": args})
        except:
            pass
    return out

tools = extract_tools(text)
log("Extracted tools", tools)
print("\n解析工具：", tools)

if tools:
    for t in tools:
        res = FUNCTION_MAPPING[t["name"]](**t["args"])
        log(f"Tool {t['name']} result", res)
        print("\n执行结果：", res)

log("=== ALL DONE ===")