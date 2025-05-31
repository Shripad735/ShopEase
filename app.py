# app.py
import streamlit as st

# Set page config FIRST, before any other Streamlit commands
st.set_page_config(
    page_title="ShopEase AI Chatbot", 
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from groq import Groq
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import time
import re

# Import pymongo for dummy MongoDB connection (add try-except for robustness)
try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None # Set to None if pymongo is not installed
    st.warning("`pymongo` not found. Install with `pip install pymongo` to enable dummy MongoDB connection features. Continuing without it.")


# Import configurations and prompts
from config import GROQ_API_KEY, GROQ_MODEL_NAME
from prompts import SYSTEM_PROMPT, PRODUCT_DATA_INSTRUCTION, ORDER_DATA_INSTRUCTION, QUICK_ACTIONS

# Custom CSS for modern aesthetic with improved performance and chat auto-scroll
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        animation: fadeIn 0.5s ease-in;
        margin-bottom: 1rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .voice-controls {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .chat-input {
        border-radius: 25px;
        border: 2px solid #667eea;
    }
    
    /* Ensure chat stays at bottom */
    .element-container:has([data-testid="stChatMessage"]) {
        scroll-margin-bottom: 100px; /* Adjust as needed */
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq client. Please check your API key: {e}")
    st.stop()

mongo_client = None
mongo_db = None
if MongoClient:
    try:
        # Replace with your actual MongoDB connection string if running a local instance
        mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000) 
        mongo_db = mongo_client["ShopEase_Chatbot"] 
        st.info("")
    except Exception as e:
        st.warning(f"Could not connect to dummy MongoDB at `mongodb://localhost:27017/`: {e}. Continuing without it.")
        mongo_client = None


# Load data functions
@st.cache_data
def load_product_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "data", "products.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("products.json not found. Make sure it's in the 'data/' directory.")
        return []
    except json.JSONDecodeError:
        st.error("Error decoding products.json. Please check file format.")
        return []

@st.cache_data
def load_order_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "data", "orders.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("orders.json not found. Creating sample order data.")
        return []
    except json.JSONDecodeError:
        st.error("Error decoding orders.json. Please check file format.")
        return []

products = load_product_data()
orders = load_order_data()

# Prepare data for LLM
product_info_for_llm = json.dumps(products, indent=2)
order_info_for_llm = json.dumps(orders, indent=2)

system_prompt_with_data = (
    SYSTEM_PROMPT + "\n\n" + 
    PRODUCT_DATA_INSTRUCTION.format(product_data=product_info_for_llm) + "\n\n" +
    ORDER_DATA_INSTRUCTION.format(order_data=order_info_for_llm)
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Hello! I'm ShopEase AI Assistant. I can help you with:\n\n🛒 Order tracking & status\n🔄 Returns & exchanges\n💰 Refunds & payments\n📦 Delivery information\n🛍️ Product recommendations\n\nHow can I assist you today?"
    })

if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False

if "processing" not in st.session_state:
    st.session_state.processing = False

if "quick_action_trigger" not in st.session_state:
    st.session_state.quick_action_trigger = None

if "auto_scroll" not in st.session_state:
    st.session_state.auto_scroll = False # Initialize to False, set to True when scroll is needed

# --- Helper Functions ---

def detect_language(text):
    """Detect if text is in Hindi or English"""
    # Check for Devanagari script
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(text):
        return "hi"
    
    # Check for common Hindi words in Roman script
    hindi_words = ['mera', 'kya', 'kahan', 'kaise', 'hai', 'mein', 'ka', 'ki', 'ko', 'aur', 'order', 'karna', 'chahta', 'chahte']
    text_lower = text.lower()
    
    # Count Hindi vs English indicators
    hindi_indicators = 0
    english_indicators = 0
    
    for word in hindi_words:
        if word in text_lower:
            hindi_indicators += 1
    
    # Common English words
    english_words = ['the', 'and', 'or', 'but', 'what', 'how', 'where', 'when', 'why', 'is', 'are', 'can', 'do', 'does', 'will', 'would', 'should', 'could']
    for word in english_words:
        if word in text_lower:
            english_indicators += 1
    
    # If more Hindi indicators, return Hindi, otherwise English
    return "hi" if hindi_indicators > english_indicators else "en"

def text_to_speech_js(text):
    """Generate JavaScript for text-to-speech"""
    # Clean text for speech
    clean_text = text.replace('\n', ' ').replace('*', '').replace('#', '').replace('`', '')
    
    js_code = f"""
    <script>
    function speakText() {{
        const text = `{clean_text}`;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.8;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        speechSynthesis.speak(utterance);
    }}
    speakText();
    </script>
    """
    return js_code

def scroll_to_bottom_js():
    """JavaScript to scroll to bottom smoothly"""
    return """
    <script>
    function scrollToBottom() {
        setTimeout(function() {
            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });
        }, 100);
    }
    scrollToBottom();
    </script>
    """

def generate_response(user_input):
    """Generate AI response with improved error handling"""
    try:
        # Construct messages for LLM with enhanced context
        llm_messages = [{"role": "system", "content": system_prompt_with_data}]
        
        # Add recent context (last 8 messages to manage token limits)
        recent_messages = st.session_state.messages[-8:]
        for message in recent_messages:
            llm_messages.append({"role": message["role"], "content": message["content"]})
        
        # Add current user input
        llm_messages.append({"role": "user", "content": user_input})

        # Call Groq API
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=llm_messages,
            stream=False,
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"😔 I apologize, but I encountered an error: {e}. Please try again or contact human support."

def generate_response_stream(api_messages_payload): # Modified to accept messages
    """Generate AI response and stream it."""
    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=api_messages_payload,
            stream=True,
            temperature=0.7,
            max_tokens=800
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"😔 I apologize, but I encountered an error: {e}."

def add_quick_action_buttons_streamlit(response_text, user_lang="en", message_index=0):
    """Add contextual quick action buttons using Streamlit buttons"""
    # Determine what buttons to show based on response content
    buttons_to_show = []
    
    if any(word in response_text.lower() for word in ['return', 'refund', 'रिटर्न']):
        if user_lang == "hi":
            buttons_to_show.extend([
                ("🔄 रिटर्न शुरू करें", "मैं एक आइटम वापस करना चाहता हूं"),
                ("📋 रिटर्न पॉलिसी", "रिटर्न पॉलिसी क्या है?")
            ])
        else:
            buttons_to_show.extend([
                ("🔄 Initiate Return", "I want to initiate a return"),
                ("📋 Return Policy", "What is your return policy?")
            ])
    
    if any(word in response_text.lower() for word in ['order', 'tracking', 'ऑर्डर']):
        if user_lang == "hi":
            buttons_to_show.append(("📦 दूसरा ऑर्डर ट्रैक करें", "मैं दूसरा ऑर्डर ट्रैक करना चाहता हूं"))
        else:
            buttons_to_show.append(("📦 Track Another Order", "I want to track another order"))
    
    if any(word in response_text.lower() for word in ['payment', 'भुगतान']):
        if user_lang == "hi":
            buttons_to_show.append(("💳 भुगतान विधियां", "भुगतान की विधियां क्या हैं?"))
        else:
            buttons_to_show.append(("💳 Payment Methods", "What payment methods do you accept?"))
    
    if any(word in response_text.lower() for word in ['product', 'उत्पाद']):
        if user_lang == "hi":
            buttons_to_show.append(("🛍️ अधिक उत्पाद", "मुझे और उत्पाद दिखाएं"))
        else:
            buttons_to_show.append(("🛍️ More Products", "Show me more products"))
    
    # Display buttons if any
    if buttons_to_show:
        st.markdown("**Quick Actions:**")
        cols = st.columns(min(len(buttons_to_show), 3))
        
        for idx, (button_text, action_text) in enumerate(buttons_to_show):
            col_idx = idx % 3
            with cols[col_idx]:
                # FIX: Add st.rerun() to quick action buttons to ensure single click functionality
                if st.button(button_text, key=f"quick_btn_{message_index}_{idx}", use_container_width=True):
                    st.session_state.quick_action_trigger = action_text
                    st.rerun() # Trigger a rerun to process the quick action immediately

def create_order_status_card(order_data):
    """Create a formatted order status card instead of timeline"""
    status_emoji = {
        "Delivered": "✅",
        "In Transit": "🚚", 
        "Processing": "⏳",
        "Refund Processing": "💰",
        "Cancelled": "❌"
    }
    
    current_status = order_data.get('status', 'Unknown')
    emoji = status_emoji.get(current_status, "📦")
    
    # Get latest tracking info
    tracking_status = order_data.get('tracking_status', [])
    latest_update = tracking_status[-1] if tracking_status else {}
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 15px 0; display: flex; align-items: center;">
            {emoji} Order {order_data['order_id']}
        </h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
            <div>
                <strong>Status:</strong><br>
                <span style="font-size: 1.1em;">{current_status}</span>
            </div>
            <div>
                <strong>Total:</strong><br>
                <span style="font-size: 1.1em;">₹{order_data.get('total_amount', 0)}</span>
            </div>
        </div>
        
        {f'''<div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 10px; margin-top: 10px;">
            <strong>Latest Update:</strong><br>
            📍 {latest_update.get('location', 'N/A')} - {latest_update.get('date', 'N/A')}
        </div>''' if latest_update else ''}
        
        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.9;">
            Order Date: {order_data.get('order_date', 'N/A')} | 
            Tracking: {order_data.get('tracking_number', 'N/A')}
        </div>
    </div>
    """
    
    return card_html

# --- UI Layout ---

# Header
st.markdown("""
<div class="main-header">
    <h1>🛍️ ShopEase AI Customer Support</h1>
    <p>Your intelligent assistant for seamless e-commerce experience</p>
</div>
""", unsafe_allow_html=True)

# --- Process New User Input (from chat_input or quick_action_trigger) ---
new_user_prompt_content = None

# Check for quick action trigger first
if st.session_state.quick_action_trigger and not st.session_state.processing:
    new_user_prompt_content = st.session_state.quick_action_trigger
    st.session_state.quick_action_trigger = None  # Consume the trigger

# Then check for typed chat input
typed_prompt = st.chat_input("💬 Ask me anything about your order, products, or services...", disabled=st.session_state.processing, key="chat_input_main")
if typed_prompt and not st.session_state.processing:
    new_user_prompt_content = typed_prompt

if new_user_prompt_content:
    st.session_state.messages.append({"role": "user", "content": new_user_prompt_content})
    st.session_state.processing = True
    # If the prompt comes from quick action or voice, it's processed here,
    # and setting `processing` to True will cause the bot response to generate
    # in the next part of the script execution.

# Main chat area
col1, col2 = st.columns([3, 1])

with col1:
    # Voice controls
    st.markdown("""
    <div class="voice-controls">
        <h3>🎤 Voice Features</h3>
        <p>Use voice input and audio responses for a hands-free experience!</p>
    </div>
    """, unsafe_allow_html=True)
    
    voice_input_enabled = st.checkbox("🎤 Enable Voice Input", value=st.session_state.voice_enabled, key="voice_input_checkbox")
    if voice_input_enabled != st.session_state.voice_enabled:
        st.session_state.voice_enabled = voice_input_enabled
        st.rerun() # Rerun if voice toggle changes, this is a minor UI change.

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and i > 0: # For non-initial assistant messages
                # TTS button
                if st.button(f"🔊 Listen", key=f"tts_{i}"):
                    st.components.v1.html(text_to_speech_js(message["content"]), height=0)
                
                # Quick action buttons for the very last assistant message
                if i == len(st.session_state.messages) - 1 and not st.session_state.processing:
                    user_lang = "en"
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        user_lang = detect_language(st.session_state.messages[i-1]["content"])
                    add_quick_action_buttons_streamlit(message["content"], user_lang, i)

    # --- Bot Response Generation and Streaming ---
    if st.session_state.processing and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Thinking..."): # Add a spinner while processing
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Typing... ⏳") 
                full_response = ""
                
                # Prepare messages for API
                api_messages_payload = [{"role": "system", "content": system_prompt_with_data}]
                # Send recent history including the latest user message
                # Using [-9:] to ensure system prompt + 8 previous user/assistant messages + current user message
                history_plus_current = st.session_state.messages[-8:] 
                for msg_content in history_plus_current:
                     api_messages_payload.append({"role": msg_content["role"], "content": msg_content["content"]})
                # Add the current user query explicitly if it's not already in history_plus_current
                if st.session_state.messages[-1]["role"] == "user" and st.session_state.messages[-1] not in history_plus_current:
                     api_messages_payload.append({"role": "user", "content": st.session_state.messages[-1]["content"]})


                try:
                    for content_chunk in generate_response_stream(api_messages_payload):
                        full_response += content_chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response) # Final response
                except Exception as e:
                    full_response = f"😔 I apologize, but I encountered an error: {e}."
                    message_placeholder.error(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.processing = False
                st.session_state.auto_scroll = True 
                st.rerun() # Only rerun AFTER bot response is complete to update UI and process next actions

with col2:
    # Sidebar information
    st.markdown("### 📊 Quick Stats")
    
    # Product stats
    total_products = len(products)
    in_stock_products = len([p for p in products if p.get('in_stock', False)])
    
    st.metric("Total Products", total_products)
    st.metric("In Stock", in_stock_products)
    st.metric("Categories", len(set(p['category'] for p in products)))
    
    # Order stats
    if orders:
        st.metric("Total Orders", len(orders))
        pending_orders = len([o for o in orders if o['status'] in ['Processing', 'In Transit']])
        st.metric("Active Orders", pending_orders)

    # Quick actions in sidebar - set trigger, don't rerun here
    st.markdown("### 🚀 Quick Actions")
    # FIX: Add st.rerun() to sidebar quick action buttons to ensure single click functionality
    if st.button("📦 Track Order", disabled=st.session_state.processing, key="qa_track_order"):
        st.session_state.quick_action_trigger = "I want to track my order"
        st.rerun() # Trigger a rerun to process the quick action immediately
    
    if st.button("🛍️ Product Info", disabled=st.session_state.processing, key="qa_prod_info"):
        st.session_state.quick_action_trigger = "Show me popular products"
        st.rerun() # Trigger a rerun to process the quick action immediately

    if st.button("🔄 Return Item", disabled=st.session_state.processing, key="qa_return_item"):
        st.session_state.quick_action_trigger = "I want to return an item"
        st.rerun() # Trigger a rerun to process the quick action immediately

    if st.button("💳 Payment Help", disabled=st.session_state.processing, key="qa_payment_help"):
        st.session_state.quick_action_trigger = "What payment methods do you accept?"
        st.rerun() # Trigger a rerun to process the quick action immediately

# Auto-scroll to bottom if needed
if st.session_state.auto_scroll:
    st.components.v1.html(scroll_to_bottom_js(), height=0)
    st.session_state.auto_scroll = False # Reset scroll trigger

# Removed the automatic order card display section as per previous request.

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown("### 🎯 Testing Scenarios")
    
    test_queries = [
        "What is the status of order ORD12345?",
        "Tell me about the Smartwatch Pro X", 
        "How do I return the Bluetooth headphones?",
        "Show me electronics products",
        "What's the refund status for order ORD12348?",
        "मेरा ऑर्डर ORD12346 कहाँ है?",
        "Which products have the best ratings?",
        "Can I change my delivery address?", 
        "What payment methods do you accept?",
        "Show me products under ₹100"
    ]
    
    for query_idx, query_text in enumerate(test_queries): # Use enumerate for unique keys
        # FIX: Add st.rerun() to test query buttons to ensure single click functionality
        if st.button(query_text, key=f"test_query_{query_idx}", disabled=st.session_state.processing):
            st.session_state.quick_action_trigger = query_text
            st.rerun() # Trigger a rerun to process the quick action immediately

    st.markdown("---")
    st.markdown("### 🔧 Features")
    st.markdown("""
    ✅ **Voice Input/Output**
    ✅ **Order Status Cards (Textual)**
    ✅ **Multi-language Support**
    ✅ **Product Recommendations**
    ✅ **Smart Context Memory**
    ✅ **Interactive UI**
    ✅ **Quick Action Buttons**
    ✅ **Streaming Responses**
    ✅ **Single-click Quick Actions**
    ✅ **Dummy MongoDB Connection**
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Use specific order IDs (ORD12345) for accurate tracking
    - Ask about product features, ratings, and availability
    - Try voice input for hands-free interaction
    - Use quick action buttons for faster responses
    - Language will match your input (Hindi/English)
    """)