#import pdb
#pdb.set_trace() 

# ============================
# 内存监控工具（核心新增！）
# ============================
import os
import gc
import psutil
import torch

def get_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)
    return round(mem, 2)

last_mem = get_memory_usage()
def mem_log(tag):
    global last_mem
    current = get_memory_usage()
    delta = current - last_mem
    print(f"[MEMORY] {tag} | {current} MB │ ↑{delta:.2f} MB")
    last_mem = current

# ============================
# 强制最低内存模式（防二次加载）
# ============================
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
if not hasattr(torch, "float8_e5m2fnuz"):
    setattr(torch, "float8_e5m2fnuz", torch.float32)

mem_log("初始启动")

# ============================
# 导入库
# ============================
from datetime import datetime
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch.optim as optim

mem_log("导入所有库完成")

# ============================
# 原版 DEBUG 日志工具（100% 还原）
# ============================
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

open(DEBUG_LOG, "w").close()
log("=== DEBUG START ===")

# ============================
# 配置
# ============================
max_seq_length = 64
model_name = "./functiongemma-270m-it"
log("CONFIG", {"max_seq_length": max_seq_length, "model_name": model_name})

# ============================
# 加载分词器
# ============================
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token
mem_log("加载 tokenizer 完成")

# ============================
# 模型只加载一次，彻底防 Kill
# ============================
model = Gemma3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager",
)
mem_log("加载模型完成（仅一次）")

# ============================
# 原版工具定义（100% 还原）
# ============================
def get_weather(city: str): return f"{city} 晴 22~28℃"
def calculate(a,b,op): return {"+":a+b,"-":a-b,"*":a*b,"/":a/b if b!=0 else 0}[op]
def date_calc(t, o): from datetime import datetime, timedelta; return (datetime.strptime(t,"%Y-%m-%d")+timedelta(o)).strftime("%Y-%m-%d")
FUNCTION_MAPPING = {"get_weather":get_weather,"calculate":calculate,"date_calc":date_calc}
TOOLS = [{
    "name":"get_weather","description":"","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
    {"name":"calculate","description":"","parameters":{"type":"object","properties":{"a":{},"b":{},"op":{}},"required":["a","b","op"]}},
    {"name":"date_calc","description":"","parameters":{"type":"object","properties":{"target_date":{},"offset_days":{}},"required":["target_date","offset_days"]}}]
log("TOOLS", TOOLS)

# ============================
# ✅ 原版完整 train_data（100% 还原）
# ============================
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

log("train_data count", len(train_data))
mem_log("训练数据加载完成")

# ============================
# 原版训练逻辑（100% 还原）
# ============================
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
model.train()
log("=== START TRAINING ===")
mem_log("进入训练模式")

for step in range(1):
    batch_text = tokenizer(
        [train_data[step % 2]["messages"][2]["content"]],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = batch_text["input_ids"]
    attention_mask = batch_text["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    log(f"TRAIN STEP {step} | LOSS", loss.item())

    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None and "lora" in name:
            log(f"GRAD {name}", param.grad.norm().item())
            break

    optimizer.step()
    optimizer.zero_grad()
    gc.collect()
    mem_log(f"训练 Step {step} 完成")

log("=== TRAIN DONE ===")

# ============================
# ✅ 原版推理 + 全部 DEBUG（100% 还原）
# ============================
model.eval()
messages = [{"role":"system","content":"使用工具回答问题"},{"role":"user","content":"2025-01-01加10天是哪天？"}]
log("Input query", messages)

inputs = tokenizer(
    tokenizer.apply_chat_template(messages, tokenize=False),
    return_tensors="pt",
    max_length=64,
    truncation=True
)
log("Inputs shape", inputs["input_ids"].shape)
mem_log("推理输入编码完成")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
    mem_log("推理前向完成")

# --------------------------
# ✅ 你所有原版 debug 100% 恢复
# --------------------------
logits = outputs.logits
log("LOGITS SHAPE", logits.shape)
log("LOGITS SAMPLE", logits[0, :8, :6])

last_hidden = outputs.hidden_states[-1]
log("FFN LAST HIDDEN SHAPE", last_hidden.shape)
log("FFN HIDDEN SAMPLE", last_hidden[0, :5, :8])  # <--- 你要的核心

softmax = torch.softmax(logits[:, -1, :], dim=-1)
log("SOFTMAX SHAPE", softmax.shape)
log("SOFTMAX TOP5", softmax.topk(5).values)

# QKV
for name, param in model.named_parameters():
    if any(x in name for x in ["q_proj", "k_proj", "v_proj"]):
        log(f"QKV WEIGHT: {name}", param.shape)
        log(f"QKV VALUE: {name}", param[:4, :4])
        break

# Layernorm
for name, module in model.named_modules():
    if "norm" in name and hasattr(module, "weight"):
        log(f"LAYERNORM: {name}", module.weight[:8])
        break

# 生成
gen_output = model.generate(**inputs, max_new_tokens=64)
text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
log("FINAL OUTPUT", text)
print("\n输出：", text)

log("=== ALL DONE ===")
mem_log("全部结束")