# LLM-Finetuning
LLM 大模型微调实践仓库，包含 LoRA 微调、Function Call 训练、CPU 低内存训练等实战项目。

---

## 子项目 1：functiongemma-tuning-lab
基于 **Gemma3 270M** 的函数调用（Function Call）LoRA 微调项目。

### 功能
- 日期计算工具调用微调
- CPU / WSL 环境可运行
- 极小内存占用（≤1.4GB）
- LoRA 轻量微调（仅训练 0.01% 参数）
- 训练 → 保存 → 测试 全流程

### 环境
- Ubuntu / WSL2
- Python 3.12
- CPU 训练（无需GPU）

### 运行
```bash
cd functiongemma-tuning-lab
python run_functiongemma_local_cpu_saved.py
