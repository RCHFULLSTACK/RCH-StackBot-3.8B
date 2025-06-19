#!/usr/bin/env python3
"""
RCH-StackBot Streamlit Web GUI
Webbaserat gränssnitt för svensk AI-assistent
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os


st.markdown("""
<style>
    /* Huvudområde - svart text på vit bakgrund */
    .stApp .main .block-container {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Sidebar - vit text på mörk bakgrund */
    .css-1d391kg, .stSidebar {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    
    .stSidebar * {
        color: #ffffff !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar div, .stSidebar span {
        color: #ffffff !important;
    }
    
    /* Chat-meddelanden - svart text */
    .stChatMessage {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage * {
        color: #000000 !important;
    }
    
    /* Huvudinnehåll - svart text */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, 
    .main p, .main div, .main span, .main label, .main .stMarkdown {
        color: #000000 !important;
    }
    
    /* Knappar i sidebar - vit text */
    .stSidebar .stButton > button {
        background-color: #1f4e79 !important;
        color: #ffffff !important;
        border: none;
        border-radius: 5px;
    }
    
    .stSidebar .stButton > button:hover {
        background-color: #2d5a3d !important;
        color: #ffffff !important;
    }
    
    /* Knappar i huvudområde - svart text */
    .main .stButton > button {
        background-color: #0066cc !important;
        color: #ffffff !important;
        border: none;
        border-radius: 5px;
    }
    
    /* Sliders och inputs i sidebar */
    .stSidebar .stSlider > div > div > div > div {
        color: #ffffff !important;
    }
    
    /* Metrics */
    .stMetric {
        color: #000000 !important;
    }
    
    /* Chat input */
    .stChatInput {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Konfigurera sidan
st.set_page_config(
    page_title="RCH-StackBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cachad modell-laddning för bättre prestanda
@st.cache_resource
def load_model():
    """Ladda AI-modellen (cachad för prestanda)"""
    with st.spinner("🚀 Laddar RCH-StackBot..."):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Försök ladda fine-tunad version
        model_path = "./models/RCH-StackBot-3.8B"
        if os.path.exists(f"{model_path}/adapter_config.json"):
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)
                return model, tokenizer, "Fine-tunad"
            except:
                pass
        
        return model, tokenizer, "Basmodell"

def generate_response(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """Generera svar från modellen"""
    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            top_p=0.9,
            top_k=50
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    if "<|end|>" in response:
        response = response.split("<|end|>")[0].strip()
    
    return response

# Ladda modell
try:
    model, tokenizer, model_type = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Fel vid laddning av modell: {e}")
    model_loaded = False

# Huvudlayout
st.title("🤖 RCH-StackBot")
st.subheader("Svensk AI-Assistent för Fullstack-Utveckling")

# Sidebar med inställningar
with st.sidebar:
    st.header("⚙️ Inställningar")
    
    if model_loaded:
        st.success(f"✅ Modell laddad: {model_type}")
        device = "🔥 CUDA" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"Enhet: {device}")
    else:
        st.error("❌ Modell ej laddad")
    
    st.subheader("🎛️ Generation")
    temperature = st.slider("Kreativitet", 0.1, 1.0, 0.7, 0.1)
    max_length = st.slider("Max längd", 50, 400, 200, 25)
    
    st.subheader("💡 Exempel-frågor")
    example_questions = [
        "Vad är React?",
        "Hur skapar jag en React component?",
        "Förklara JavaScript closures",
        "Vad är skillnaden mellan let och const?",
        "Hur fungerar async/await?",
        "Hur skapar jag en Express server?",
        "Vad är en REST API?",
        "Förklara CSS flexbox"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{question}", use_container_width=True):
            st.session_state.current_question = question

# Huvudinnehåll
if model_loaded:
    # Chat-historik
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Visa välkomstmeddelande
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.write("""
            👋 **Hej! Jag är RCH-StackBot, din svenska AI-assistent för webbutveckling!**
            
            Jag kan hjälpa dig med:
            • **Frontend**: React, Vue, Angular, HTML, CSS
            • **Backend**: Node.js, Express, Python, databaser
            • **JavaScript**: ES6+, TypeScript, async/await
            • **DevOps**: Docker, deployment, API:er
            
            Ställ en fråga eller välj ett exempel från sidomenyn! 🚀
            """)
    
    # Visa chat-historik
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input från användare
    user_input = None
    
    # Kolla om det finns en förvald fråga
    if "current_question" in st.session_state:
        user_input = st.session_state.current_question
        del st.session_state.current_question
    else:
        user_input = st.chat_input("Ställ din fråga om webbutveckling här...")
    
    if user_input:
        # Lägg till användarmeddelande
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generera och visa svar
        with st.chat_message("assistant"):
            with st.spinner("🤔 Tänker..."):
                start_time = time.time()
                
                try:
                    response = generate_response(
                        model, tokenizer, user_input, 
                        max_length=max_length, 
                        temperature=temperature
                    )
                    
                    generation_time = time.time() - start_time
                    
                    st.write(response)
                    st.caption(f"⏱️ Genererad på {generation_time:.1f}s")
                    
                    # Lägg till i historik
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"❌ Fel vid generering: {e}")
    
    # Footer med statistik
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Konversationer", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    with col2:
        if st.button("🗑️ Rensa chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col3:
        st.metric("Modell", model_type)

else:
    st.error("⚠️ Kunde inte ladda AI-modellen. Kontrollera att alla dependencies är installerade.")
    
    with st.expander("🔧 Felsökningsinstruktioner"):
        st.code("""
# Installera dependencies
pip install streamlit torch transformers

# Starta appen
streamlit run streamlit_app.py
        """)

# CSS för bättre styling
st.markdown("""
<style>
    .stApp > header {
        background-color: transparent;
    }
    
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)