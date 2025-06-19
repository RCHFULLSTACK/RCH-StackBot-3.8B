#!/usr/bin/env python3
"""
RCH-StackBot-3.8B Setup Script
Automatisk installation och konfiguration
"""

import os
import sys
import subprocess
import torch

def check_python_version():
    """Kontrollera Python-version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ krävs")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

def check_gpu():
    """Kontrollera GPU-tillgänglighet"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 10:
            print("⚠️  Varning: Mindre än 10 GB GPU-minne kan orsaka problem")
    else:
        print("❌ Ingen CUDA-kompatibel GPU hittades")
        print("   Modellen kommer köras på CPU (mycket långsamt)")

def install_requirements():
    """Installera Python-paket"""
    print("\n📦 Installerar Python-paket...")
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.36.0", 
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0"
    ]
    
    for package in requirements:
        try:
            print(f"  Installerar {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"❌ Misslyckades att installera {package}")
            return False
    
    print("✅ Alla paket installerade")
    return True

def create_directories():
    """Skapa nödvändiga mappar"""
    print("\n📁 Skapar mappar...")
    
    dirs = [
        "data",
        "models", 
        "models/RCH-StackBot-3.8B",
        "scripts",
        "logs"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✅ {dir_name}")

def download_base_model():
    """Ladda ner basmodell för snabbare träning"""
    print("\n⬇️  Laddar ner basmodell...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        cache_dir = "./models/phi3-base"
        
        print(f"  Laddar tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        print(f"  Laddar modell...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16
        )
        
        print("✅ Basmodell nedladdad och cached")
        return True
        
    except Exception as e:
        print(f"❌ Misslyckades att ladda basmodell: {e}")
        return False

def create_sample_data():
    """Skapa exempeldata om ingen finns"""
    data_file = "data/training_data.txt"
    
    if not os.path.exists(data_file):
        print(f"\n📝 Skapar exempeldata: {data_file}")
        
        sample_data = """<|user|>
Hej! Vad kan du hjälpa mig med?
<|end|>
<|assistant|>
Hej! Jag är RCH-StackBot och jag kan hjälpa dig med fullstack-utveckling på svenska. Jag kan svara på frågor om:

- Frontend: React, Vue, Angular, HTML, CSS, JavaScript
- Backend: Node.js, Python, Java, API-utveckling
- Databaser: SQL, NoSQL, MongoDB, PostgreSQL
- DevOps: Docker, CI/CD, deployment
- Och mycket mer!

Vad skulle du vilja veta?
<|end|>

<|user|>
Hur skapar jag en enkel React-komponent?
<|end|>
<|assistant|>
Här är hur du skapar en enkel React-komponent:

```javascript
import React from 'react';

// Funktionskomponent
const MyComponent = () => {
  return (
    <div>
      <h1>Hej från min komponent!</h1>
      <p>Detta är en enkel React-komponent.</p>
    </div>
  );
};

export default MyComponent;
```

Du kan också skapa den med arrow function:
```javascript
const MyComponent = () => (
  <div>
    <h1>Hej från min komponent!</h1>
  </div>
);
```

För att använda komponenten:
```javascript
import MyComponent from './MyComponent';

function App() {
  return (
    <div>
      <MyComponent />
    </div>
  );
}
```
<|end|>"""
        
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        
        print("✅ Exempeldata skapad")

def main():
    print("🚀 RCH-StackBot-3.8B Setup")
    print("=" * 40)
    
    # Kontroller
    check_python_version()
    check_gpu()
    
    # Installation
    if not install_requirements():
        print("❌ Setup misslyckades vid paketinstallation")
        sys.exit(1)
    
    # Skapa mappar
    create_directories()
    
    # Ladda basmodell
    if not download_base_model():
        print("⚠️  Varning: Basmodell kunde inte laddas (kommer laddas vid träning)")
    
    # Skapa exempeldata
    create_sample_data()
    
    print("\n" + "=" * 40)
    print("🎉 Setup klar!")
    print("\nNästa steg:")
    print("1. Lägg till din träningsdata i data/training_data.txt")
    print("2. Kör: python train.py")
    print("3. Testa modellen: python inference.py --interactive")

if __name__ == "__main__":
    main()