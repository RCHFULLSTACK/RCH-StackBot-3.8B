#!/usr/bin/env python3
"""
RCH-StackBot-3.8B Inference Script
Använd den tränade modellen för att generera svar
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

class RCHStackBot:
    def __init__(self, model_path="./models/RCH-StackBot-3.8B", base_model="microsoft/Phi-3-mini-4k-instruct"):
        """Initialisera RCH-StackBot"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Använder enhet: {self.device}")
        
        # Ladda tokenizer
        print("Laddar tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ladda basmodell
        print("Laddar basmodell...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Ladda LoRA-weights om de finns
        if os.path.exists(model_path):
            print(f"Laddar fine-tunad modell från: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print("Ingen fine-tunad modell hittades. Använder basmodell.")
        
        self.model.eval()
        
    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """Generera svar på en prompt"""
        # Formatera prompt för Phi-3
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Tokenisera
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generera
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Dekoda svar
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrahera endast assistant-delen
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    def interactive_chat(self):
        """Interaktiv chat-session"""
        print("=== RCH-StackBot-3.8B Interactive Chat ===")
        print("Skriv 'quit' för att avsluta")
        print("Skriv 'clear' för att rensa historik")
        print("-" * 50)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\nDu: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Hej då!")
                    break
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("Historik rensad!")
                    continue
                
                if not user_input:
                    continue
                
                # Lägg till kontext från tidigare konversation
                context = ""
                if conversation_history:
                    context = "\n".join([f"Användare: {h['user']}\nAssistent: {h['bot']}" for h in conversation_history[-3:]])
                    context += "\n\n"
                
                full_prompt = context + user_input
                
                print("RCH-StackBot: ", end="", flush=True)
                response = self.generate_response(full_prompt)
                print(response)
                
                # Spara i historik
                conversation_history.append({
                    "user": user_input,
                    "bot": response
                })
                
            except KeyboardInterrupt:
                print("\n\nAvbrutet av användare. Hej då!")
                break
            except Exception as e:
                print(f"\nFel: {e}")
                continue

def batch_inference(bot, input_file, output_file):
    """Kör inferens på flera prompts från fil"""
    print(f"Läser prompts från: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Bearbetar prompt {i}/{len(prompts)}: {prompt[:50]}...")
        
        response = bot.generate_response(prompt)
        
        results.append({
            "prompt": prompt,
            "response": response,
            "prompt_id": i
        })
    
    # Spara resultat
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Resultat sparade till: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="RCH-StackBot-3.8B Inference")
    parser.add_argument("--model_path", default="./models/RCH-StackBot-3.8B", help="Sökväg till modell")
    parser.add_argument("--interactive", action="store_true", help="Interaktiv chat-mode")
    parser.add_argument("--input", help="Input-fil för batch-inferens")
    parser.add_argument("--output", help="Output-fil för batch-inferens")
    parser.add_argument("--prompt", help="Enskild prompt att testa")
    parser.add_argument("--max_length", type=int, default=512, help="Max längd för genererat svar")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature för sampling")
    
    args = parser.parse_args()
    
    # Initialisera bot
    bot = RCHStackBot(model_path=args.model_path)
    
    if args.interactive:
        # Interaktiv mode
        bot.interactive_chat()
    
    elif args.input and args.output:
        # Batch-inferens
        batch_inference(bot, args.input, args.output)
    
    elif args.prompt:
        # Enskild prompt
        print("Prompt:", args.prompt)
        print("Svar:", bot.generate_response(args.prompt, max_length=args.max_length, temperature=args.temperature))
    
    else:
        # Default: interaktiv mode
        bot.interactive_chat()

if __name__ == "__main__":
    main()