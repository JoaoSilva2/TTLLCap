class LoRA_Modules:
    def __init__(self) -> None:
        modules = {}

        modules["gpt2"] = ["attn.c_attn"]

        modules["opt-1.3b"] = []
        for i in range(24):
                modules["opt-1.3b"].append(f"decoder.layers.{i}.self_attn.k_proj")
                modules["opt-1.3b"].append(f"decoder.layers.{i}.self_attn.q_proj")
                modules["opt-1.3b"].append(f"decoder.layers.{i}.self_attn.v_proj")
    
    def get_modules_from_name(self, decoder_name):
        for k, v in self.modules.items():
            if k in decoder_name:
                return v