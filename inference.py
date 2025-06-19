#!/usr/bin/env python3
"""
RCH-StackBot-3.8B - Final Production Version
Svensk Fullstack Utvecklingsassistent
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time
from datetime import datetime

class RCHStackBot:
    def __init__(self, model_path="./models/RCH-StackBot-3.8B"):
        print("🚀 RCH-StackBot startar...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Enhet: {self.device}")
        
        # Ladda modell
        self._load_model(model_path)
        print("✅ RCH-StackBot är redo!\n")
    
    def _load_model(self, model_path):
        """Ladda modell och tokenizer"""
        print("📥 Laddar tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("📥 Laddar basmodell...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Försök ladda fine-tunad version
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                from peft import PeftModel
                print("📥 Laddar fine-tunad modell...")
                self.model = PeftModel.from_pretrained(self.model, model_path)
                print("✅ Fine-tunad modell laddad!")
                self.is_finetuned = True
            except Exception as e:
                print(f"⚠️ Kunde inte ladda fine-tunad modell: {e}")
                print("📝 Använder basmodell")
                self.is_finetuned = False
        else:
            print("📝 Använder basmodell (ingen fine-tuning hittades)")
            self.is_finetuned = False

    def generate_response(self, prompt, max_length=200, temperature=0.7):
        """Generera svar från modellen"""
        # Enkel prompt-formatering för Phi-3
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Tokenisera
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generera svar
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                top_p=0.9,
                top_k=50
            )
        
        # Dekoda och rensa svaret
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrahera assistant-delen
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Rensa artefakter
        if "<|end|>" in response:
            response = response.split("<|end|>")[0].strip()
        
        return response

    def interactive_chat(self):
        """Interaktiv chat-session"""
        print("🤖 RCH-StackBot Interactive Chat")
        print("=" * 50)
        print("🎯 Specialområden:")
        print("   • Frontend: React, Vue, Angular, HTML, CSS")
        print("   • Backend: Node.js, Express, Python, databases")  
        print("   • JavaScript: ES6+, TypeScript, async/await")
        print("   • DevOps: Docker, deployment, API:er")
        print("   • Allmänt: Kodning, best practices, debugging")
        
        model_status = "Fine-tunad" if self.is_finetuned else "Basmodell"
        print(f"📊 Status: {model_status}")
        
        print("\n💬 Kommandon:")
        print("   • 'exit' - Avsluta")
        print("   • 'help' - Visa hjälp")
        print("   • 'exempel' - Visa exempel-frågor")
        print("=" * 50)
        
        conversation_count = 0
        start_time = time.time()
        
        while True:
            try:
                user_input = input(f"\n🧑‍💻 [{conversation_count + 1}] Du: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'avsluta', 'bye']:
                    duration = time.time() - start_time
                    print(f"\n📊 Session-statistik:")
                    print(f"   Frågor: {conversation_count}")
                    print(f"   Tid: {duration:.1f} sekunder")
                    print("👋 Tack för att du använde RCH-StackBot!")
                    break
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['help', 'hjälp']:
                    self._show_help()
                    continue
                
                if user_input.lower() in ['exempel', 'examples']:
                    self._show_examples()
                    continue
                
                # Generera svar
                print("🤖 Tänker...", end="", flush=True)
                start_gen = time.time()
                
                response = self.generate_response(user_input)
                
                gen_time = time.time() - start_gen
                print(f"\r🤖 Bot [{gen_time:.1f}s]: {response}\n")
                
                conversation_count += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat avbruten. Hej då!")
                break
            except Exception as e:
                print(f"\n❌ Fel uppstod: {e}")
                print("Försök igen...")

    def _show_help(self):
        """Visa hjälp-information"""
        print("\n🆘 RCH-StackBot Hjälp")
        print("=" * 30)
        print("📚 Bästa sätt att ställa frågor:")
        print("• Var specifik: 'Hur skapar jag en React useState hook?'")
        print("• Nämn teknik: 'I Node.js, hur läser jag en fil?'")
        print("• Be om exempel: 'Visa kod för en Express route'")
        print("• Felsökning: 'Varför får jag CORS-fel i min API?'")
        
    def _show_examples(self):
        """Visa exempel-frågor"""
        print("\n💡 Exempel-frågor:")
        print("=" * 20)
        examples = [
            "Hur skapar jag en React functional component?",
            "Vad är skillnaden mellan == och === i JavaScript?",
            "Hur sätter jag upp en Express server med CORS?",
            "Förklara async/await vs Promises",
            "Hur använder jag CSS Grid för layout?",
            "Vad är skillnaden mellan localStorage och sessionStorage?",
            "Hur gör jag en fetch-request i JavaScript?",
            "Förklara vad REST API:er är"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")

    def demo_mode(self):
        """Demo med exempel-frågor"""
        print("🎯 RCH-StackBot Demo Mode")
        print("=" * 30)
        
        demo_questions = [
            "Vad är React?",
            "Hur skapar jag en variabel i JavaScript?", 
            "Förklara vad en REST API är",
            "Vad är skillnaden mellan let och const?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{i}. 📝 {question}")
            response = self.generate_response(question, max_length=150)
            print(f"🤖 {response}")
            print("-" * 60)
        
        print(f"\n💡 Kör med --interactive för chat-mode")

def main():
    parser = argparse.ArgumentParser(
        description="RCH-StackBot - Svensk Fullstack Utvecklingsassistent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exempel:
  python stackbot.py                                    # Demo mode
  python stackbot.py --interactive                      # Chat mode  
  python stackbot.py --prompt "Vad är React?"          # Enskild fråga
  python stackbot.py --prompt "Kod exempel" --temp 0.3  # Lägre kreativitet
        """
    )
    
    parser.add_argument("--prompt", "-p", type=str, help="Enskild fråga att ställa")
    parser.add_argument("--interactive", "-i", action="store_true", help="Starta interaktiv chat")
    parser.add_argument("--model_path", "-m", default="./models/RCH-StackBot-3.8B", help="Sökväg till modell")
    parser.add_argument("--max_length", "-l", type=int, default=200, help="Max tokens i svar")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Kreativitet (0.1-1.0)")
    
    args = parser.parse_args()
    
    try:
        # Skapa bot
        bot = RCHStackBot(model_path=args.model_path)
        
        if args.prompt:
            # Enskild fråga
            print(f"📝 Fråga: {args.prompt}")
            print("🤖 Svar:")
            response = bot.generate_response(
                args.prompt, 
                max_length=args.max_length, 
                temperature=args.temperature
            )
            print(response)
            
        elif args.interactive:
            # Interaktiv mode
            bot.interactive_chat()
            
        else:
            # Demo mode
            bot.demo_mode()
    
    except KeyboardInterrupt:
        print("\n👋 Programmet avbröts. Hej då!")
    except Exception as e:
        print(f"\n❌ Ett fel uppstod: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()