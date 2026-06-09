# ======================
# 本地运行 FunctionGemma 270M
# 完整可运行代码（无Colab依赖）
# ======================

import torch
from unsloth import FastLanguageModel

# ======================
# 1. 加载模型
# ======================
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./functiongemma-270m-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ======================
# 2. LoRA 配置
# ======================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ======================
# 3. 工具定义
# ======================
def get_weather(city: str):
    return f"{city} 今日晴，气温 22~28℃"

def calculate(a: float, b: float, op: str):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b if b != 0 else 0

# 新增：日期换算工具
def date_calc(target_date: str, offset_days: int):
    """
    target_date: 基准日期,格式YYYY-MM-DD
    offset_days: 偏移天数，正数往后、负数往前
    return: 偏移后日期字符串
    """
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
        "description": "输入基准日期和偏移天数，计算偏移后的目标日期，日期格式必须为YYYY-MM-DD，offset_days正数向后推、负数向前推",
        "parameters": {
            "type": "object",
            "properties": {
                "target_date": {"type": "string", "description": "基准日期，格式YYYY-MM-DD"},
                "offset_days": {"type": "int", "description": "需要偏移的天数，整数"}
            },
            "required": ["target_date", "offset_days"]
        }
    }
]


# 追加进TOOLS列表，拼接在原有get_weather、calculate后面
{
    "name": "date_calc",
    "description": "输入基准日期和偏移天数，计算偏移后的目标日期，日期格式必须为YYYY-MM-DD，offset_days正数向后推、负数向前推",
    "parameters": {
        "type": "object",
        "properties": {
            "target_date": {"type": "string", "description": "基准日期，格式YYYY-MM-DD"},
            "offset_days": {"type": "int", "description": "需要偏移的天数，整数"}
        },
        "required": ["target_date", "offset_days"]
    }
}

# ======================
# 4. 构造训练数据
# ======================
train_data = [
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "从2025-01-01往后推15天是哪天？"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-01\",\"offset_days\":15}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-01-10往前倒退8天是什么日期"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-10\",\"offset_days\":-8}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "以2025-02-01为基准，向后加45天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-02-01\",\"offset_days\":45}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-02-15往前22天是几号"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-02-15\",\"offset_days\":-22}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "算一下2025-03-03再加60天的日期"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-03-03\",\"offset_days\":60}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-03-20向前减35天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-03-20\",\"offset_days\":-35}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "基准2025-04-05，往后推移28天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-04-05\",\"offset_days\":28}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "从2025-04-22往前推50天是哪天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-04-22\",\"offset_days\":-50}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-05-01增加90天日期"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-05-01\",\"offset_days\":90}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-05-18倒退12天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-05-18\",\"offset_days\":-12}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-06-02往后33天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-06-02\",\"offset_days\":33}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-06-25往前66天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-06-25\",\"offset_days\":-66}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "以2025-07-04为准，向后18天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-07-04\",\"offset_days\":18}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-07-30往前推42天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-07-30\",\"offset_days\":-42}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-08-08加75天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-08-08\",\"offset_days\":75}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-08-22减去29天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-08-22\",\"offset_days\":-29}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-09-01向后55天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-09-01\",\"offset_days\":55}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-09-19往前99天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-09-19\",\"offset_days\":-99}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-10-05加21天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-10-05\",\"offset_days\":21}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-10-28倒退38天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-10-28\",\"offset_days\":-38}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "帮我算2025-11-03向后推62天是哪天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-11-03\",\"offset_days\":62}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "麻烦计算2025-11-22往前17天的日期"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-11-22\",\"offset_days\":-17}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "已知基准2025-12-06，向后加49天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-12-06\",\"offset_days\":49}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-12-25往前推72天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-12-25\",\"offset_days\":-72}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-01-02往后13天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-01-02\",\"offset_days\":13}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-01-15往前59天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-01-15\",\"offset_days\":-59}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "计算2026-02-05增加81天日期"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-02-05\",\"offset_days\":81}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-02-20减去31天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-02-20\",\"offset_days\":-31}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-03-07向后39天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-03-07\",\"offset_days\":39}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-03-29往前推69天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-03-29\",\"offset_days\":-69}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "我想知道2026-04-04再加26天是什么日子"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-04-04\",\"offset_days\":26}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "帮我查2026-04-21倒退47天日期"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-04-21\",\"offset_days\":-47}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "基准日期2026-05-06，向后77天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-05-06\",\"offset_days\":77}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-05-30往前24天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-05-30\",\"offset_days\":-24}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-06-03加53天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-06-03\",\"offset_days\":53}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-06-27往前88天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-06-27\",\"offset_days\":-88}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "从2025-01-05往后推7天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-05\",\"offset_days\":7}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-01-18往前33天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-18\",\"offset_days\":-33}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-02-07向后92天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-02-07\",\"offset_days\":92}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-02-26倒退19天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-02-26\",\"offset_days\":-19}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "算一算2025-03-09向后41天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-03-09\",\"offset_days\":41}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-03-27往前54天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-03-27\",\"offset_days\":-54}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-04-11加67天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-04-11\",\"offset_days\":67}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-04-29减27天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-04-29\",\"offset_days\":-27}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-05-07向后36天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-05-07\",\"offset_days\":36}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-05-24往前77天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-05-24\",\"offset_days\":-77}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-06-11加11天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-06-11\",\"offset_days\":11}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-06-30倒退61天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-06-30\",\"offset_days\":-61}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-07-09向后83天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-07-09\",\"offset_days\":83}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-07-25往前23天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-07-25\",\"offset_days\":-23}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-08-12加48天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-08-12\",\"offset_days\":48}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-08-31往前32天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-08-31\",\"offset_days\":-32}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-09-07向后71天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-09-07\",\"offset_days\":71}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-09-23倒退14天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-09-23\",\"offset_days\":-14}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-10-11加57天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-10-11\",\"offset_days\":57}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-10-26往前82天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-10-26\",\"offset_days\":-82}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-11-09向后29天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-11-09\",\"offset_days\":29}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-11-30倒退43天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-11-30\",\"offset_days\":-43}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-12-13加79天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-12-13\",\"offset_days\":79}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-12-28往前21天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-12-28\",\"offset_days\":-21}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-01-07向后51天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-01-07\",\"offset_days\":51}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-01-22倒退94天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-01-22\",\"offset_days\":-94}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-02-09加34天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-02-09\",\"offset_days\":34}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-02-27往前64天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-02-27\",\"offset_days\":-64}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-03-12向后73天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-03-12\",\"offset_days\":73}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-03-25倒退16天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-03-25\",\"offset_days\":-16}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-04-08加44天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-04-08\",\"offset_days\":44}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-04-30往前58天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-04-30\",\"offset_days\":-58}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-05-14向后86天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-05-14\",\"offset_days\":86}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-05-27倒退28天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-05-27\",\"offset_days\":-28}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-06-09加39天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-06-09\",\"offset_days\":39}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "从2025-01-03往前推5天是哪天？"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-03\",\"offset_days\":-5}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-01-22向后推65天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-01-22\",\"offset_days\":65}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-02-11往前46天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-02-11\",\"offset_days\":-46}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-02-28加59天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-02-28\",\"offset_days\":59}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-03-14往后22天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-03-14\",\"offset_days\":22}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-03-31往前74天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-03-31\",\"offset_days\":-74}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-04-16加31天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-04-16\",\"offset_days\":31}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-05-02倒退85天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-05-02\",\"offset_days\":-85}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-05-19向后19天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-05-19\",\"offset_days\":19}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-06-07往前40天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-06-07\",\"offset_days\":-40}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-06-22加70天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-06-22\",\"offset_days\":70}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-07-13倒退37天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-07-13\",\"offset_days\":-37}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-07-28向后52天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-07-28\",\"offset_days\":52}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-08-16往前91天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-08-16\",\"offset_days\":-91}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-08-29加25天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-08-29\",\"offset_days\":25}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-09-12倒退55天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-09-12\",\"offset_days\":-55}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-09-27向后78天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-09-27\",\"offset_days\":78}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-10-17往前10天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-10-17\",\"offset_days\":-10}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-11-02加42天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-11-02\",\"offset_days\":42}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-11-18倒退68天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-11-18\",\"offset_days\":-68}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-12-04向后30天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-12-04\",\"offset_days\":30}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2025-12-22往前80天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2025-12-22\",\"offset_days\":-80}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-01-11加63天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-01-11\",\"offset_days\":63}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-01-27倒退20天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-01-27\",\"offset_days\":-20}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-02-13向后47天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-02-13\",\"offset_days\":47}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-02-24往前76天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-02-24\",\"offset_days\":-76}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-03-18加32天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-03-18\",\"offset_days\":32}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-04-02倒退50天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-04-02\",\"offset_days\":-50}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-04-17向后89天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-04-17\",\"offset_days\":89}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-05-09倒退34天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-05-09\",\"offset_days\":-34}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-05-22加56天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-05-22\",\"offset_days\":56}<end_function_call>"}]},
    {"messages": [{"role": "system", "content": "使用工具回答问题"},{"role": "user", "content": "2026-06-01往前7天"},{"role": "assistant", "content": "<start_function_call>call:date_calc{\"target_date\":\"2026-06-01\",\"offset_days\":-7}<end_function_call>"}]},
]

from datasets import Dataset
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

# ======================
# 5. 开始训练
# ======================
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    training_arguments=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=2,
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=3407,
    ),
)

trainer.train()

# ======================
# 6. 推理测试（本地可用）
# ======================
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "使用工具回答问题"},
    {"role": "user", "content": "上海天气？"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tools=TOOLS,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n模型输出：\n", text)

# ======================
# 7. 提取函数调用
# ======================
import re, json

def extract_tools(text):
    pattern = r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    out = []
    for name, argstr in matches:
        try:
            args = json.loads("{" + argstr + "}")
            out.append({"name": name, "args": args})
        except: pass
    return out

tools = extract_tools(text)
print("\n解析工具调用：", tools)

# ======================
# 8. 执行工具并输出最终结果
# ======================
if tools:
    for tool in tools:
        func = FUNCTION_MAPPING[tool["name"]]
        result = func(**tool["args"])
        print("\n工具执行结果：", result)