    def run_inference(self, prompt: str) -> str:
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
