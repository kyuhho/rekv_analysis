import sys
from unittest.mock import MagicMock

# Mock transformers and other dependencies
sys.modules['transformers'] = MagicMock()
sys.modules['transformers.models.qwen2.modeling_qwen2'] = MagicMock()
sys.modules['logzero'] = MagicMock()
sys.modules['model.attention'] = MagicMock()
sys.modules['model.patch'] = MagicMock()

# Mock specific classes
class MockConfig:
    def __init__(self):
        self.num_hidden_layers = 32
        self.vision_feature_layer = -1
        self.vision_feature_select_strategy = "default"
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_cache = True
        self.use_return_dict = True

class MockAttention:
    def __init__(self):
        self.forward = lambda *args, **kwargs: None
        self.rotary_emb = MagicMock()
        self.rotary_emb.config.rope_theta = 10000.0

class MockLayer:
    def __init__(self):
        self.self_attn = MockAttention()

class MockModel:
    def __init__(self):
        self.layers = [MockLayer() for _ in range(32)]
        self.config = MockConfig()
        self.position_bias = None
        self.norm = MagicMock()
        self.embed_tokens = MagicMock()

    def apply(self, fn):
        for layer in self.layers:
            fn(layer.self_attn)

class MockLlavaOnevisionForConditionalGeneration:
    def __init__(self, config):
        self.config = config
        self.language_model = MockModel()
        self.language_model.model = self.language_model # Hack for patch_hf accessing model.model
        self.vision_tower = MagicMock()
        self.multi_modal_projector = MagicMock()
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(MockConfig())
    
    def eval(self):
        pass

sys.modules['transformers'].LlavaOnevisionForConditionalGeneration = MockLlavaOnevisionForConditionalGeneration
sys.modules['transformers'].LlavaOnevisionProcessor = MagicMock()

# Import the actual modules to test
# We need to unmock model.patch to test the actual patch_hf logic, 
# but we need to mock rekv_attention_forward to avoid importing real attention code which might depend on torch/cuda
import model.patch
from model.patch import patch_hf

# Mock rekv_attention_forward to capture arguments
def mock_rekv_attention_forward(**kwargs):
    def forward(self, *args, **kwargs2):
        return None, None
    forward.kwargs = kwargs
    return forward

model.patch.rekv_attention_forward = mock_rekv_attention_forward

# Now import load_model
from model.llava_onevision_rekv import load_model

def verify():
    print("Verifying budget allocation...")
    topk = 64
    model, _ = load_model(topk=topk)
    
    layers = model.language_model.model.layers
    
    correct = True
    for i, layer in enumerate(layers):
        # The forward method is now a bound method of the closure returned by rekv_attention_forward
        # We can access the closure's captured variables via __closure__ if it was a real function,
        # but here we mocked rekv_attention_forward to return a function with .kwargs attribute.
        # However, patch_hf wraps this in hugginface_forward.
        
        # Let's inspect how patch_hf wraps it.
        # forward = huggingface_forward(rekv_attention_forward(**kwargs_i))
        # m.forward = forward.__get__(m, Attention)
        
        # We need to dig into the wrapped function.
        # huggingface_forward returns hf_forward which calls 'forward' (the result of rekv_attention_forward)
        
        # Since we cannot easily unwrap the closure of huggingface_forward without more hacks,
        # let's modify the mock_rekv_attention_forward to print/store the topk it received.
        pass

    # Actually, let's just inspect the topk_list passed to patch_hf?
    # No, we want to verify the end result on the layers.
    
    # In patch_hf:
    # forward_i = huggingface_forward(rekv_attention_forward(**kwargs_i))
    # m.forward = forward_i.__get__(m, Attention)
    
    # So m.forward is a bound method.
    # m.forward.__func__ is hf_forward.
    # hf_forward is a closure capturing 'forward' (from rekv_attention_forward).
    
    # It's hard to inspect closures in a robust way across python versions without 'inspect'.
    # But we can verify by side effect.
    
    # Let's redefine mock_rekv_attention_forward to store the topk in a global list
    pass

captured_topks = []
def capturing_rekv_attention_forward(**kwargs):
    captured_topks.append(kwargs.get('topk'))
    def forward(*args, **kwargs): return None
    return forward

model.patch.rekv_attention_forward = capturing_rekv_attention_forward

# Reload to capture
print("Reloading model to capture topk...")
captured_topks.clear()
model, _ = load_model(topk=64)

expected_high = int(64 * 1.5) # 96
expected_low = int(64 * 0.5)  # 32

print(f"Captured {len(captured_topks)} layers.")
if len(captured_topks) != 32:
    print("Error: Expected 32 layers.")
    correct = False
else:
    for i in range(16):
        if captured_topks[i] != expected_high:
            print(f"Layer {i}: Expected {expected_high}, got {captured_topks[i]}")
            correct = False
    for i in range(16, 32):
        if captured_topks[i] != expected_low:
            print(f"Layer {i}: Expected {expected_low}, got {captured_topks[i]}")
            correct = False

if correct:
    print("SUCCESS: Budget allocation is correct (3:1 split).")
else:
    print("FAILURE: Budget allocation is incorrect.")

if __name__ == "__main__":
    verify()
