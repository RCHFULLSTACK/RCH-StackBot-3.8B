#!/usr/bin/env python3
"""
Enkel test av RCH-StackBot-3.8B
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def test_model():
    print("🚀 Testar RCH-StackBot-3.8B...")
    
    # Ladda basmodell
    print("📥 Laddar basmodell...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # Fixa attention-problemet
    )
    
    # Ladda fine-tunad modell om den finns
    model_path = "./models/RCH-StackBot-3.8B"
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print("🎯 Laddar fine-tunad modell...")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("⚠️ Ingen fine-tunad modell hittades, använder basmodell")
        model = base_model
    
    # Test prompts
    test_prompts = [
        "Förklara vad React är på svenska",
        "Hur skapar jag en enkel Node.js server?",
        "Vad är skillnaden mellan let och const i JavaScript?"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("🤖 Svar:", end=" ")
        
        # Formatera för Phi-3
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Tokenisera
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to("cuda")
        
        # Generera med simpla inställningar
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Dekoda svaret
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extrahera assistant-delen
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            print(response)
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Fel: {e}")
            continue

if __name__ == "__main__":
    test_model()