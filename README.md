# ShopEase AI Customer Support Chatbot 🛍️

A modern, AI-powered customer support chatbot for e-commerce platforms built with Streamlit and Groq AI. This intelligent assistant helps customers with order tracking, product information, returns, refunds, and more.

## 🌟 Features

- **Multi-language Support** - Supports both English and Hindi (Devanagari script)
- **Order Tracking** - Real-time order status and tracking information
- **Product Recommendations** - Smart product suggestions with ratings and reviews
- **Voice Features** - Text-to-speech responses for accessibility
- **Quick Actions** - Contextual buttons for faster interactions
- **Streaming Responses** - Real-time AI response generation
- **Interactive UI** - Modern gradient design with smooth animations
- **Smart Context Memory** - Maintains conversation context across interactions

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API Key (free from [Groq Console](https://console.groq.com/))

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd ecommerce-chatbot
```
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a [`.env`](ecommerce-chatbot/.env) file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Verify Data Files
Ensure the following data files are present in the [`data/`](ecommerce-chatbot/data) directory:
- [`products.json`](ecommerce-chatbot/data/products.json) - Product catalog
- [`orders.json`](ecommerce-chatbot/data/orders.json) - Sample order data

## 🏃‍♂️ Running the Application

### Start the Streamlit App
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Alternative Port (if 8501 is busy)
```bash
streamlit run app.py --server.port 8502
```

## 📁 Project Structure

```
ecommerce-chatbot/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── prompts.py            # AI prompts and templates
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore rules
├── data/
│   ├── products.json    # Product catalog data
│   └── orders.json      # Sample order data
└── __pycache__/         # Python cache files
```

## 🔧 Configuration

### Environment Variables
Edit [`.env`](ecommerce-chatbot/.env) file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### AI Model Settings
Modify [`config.py`](ecommerce-chatbot/config.py) to change the AI model:
```python
GROQ_MODEL_NAME = "llama3-8b-8192"  # Default model
# Other options: "mixtral-8x7b-32768", "llama3-70b-8192"
```

### Customizing Prompts
Edit [`prompts.py`](ecommerce-chatbot/prompts.py) to customize:
- System prompts for AI behavior
- Language-specific responses
- Quick action templates

## 🧪 Testing the Chatbot

### Sample Test Queries
Try these example queries to test functionality:

**Order Tracking:**
- "What is the status of order ORD12345?"
- "मेरा ऑर्डर ORD12346 कहाँ है?" (Hindi)

**Product Information:**
- "Tell me about the Smartwatch Pro X"
- "Show me electronics products"
- "Which products have the best ratings?"

**Returns & Refunds:**
- "How do I return the Bluetooth headphones?"
- "What's the refund status for order ORD12348?"

**General Support:**
- "What payment methods do you accept?"
- "Can I change my delivery address?"

### Testing Features
1. **Voice Features** - Toggle voice input checkbox
2. **Quick Actions** - Use sidebar buttons for common queries
3. **Multi-language** - Switch between English and Hindi inputs
4. **Streaming** - Watch responses generate in real-time

## 🛠️ Troubleshooting

### Performance Tips
- The app caches product and order data for better performance
- Clear cache in Streamlit if data files are updated
- Use Ctrl+C to stop the development server

## 🔒 Security Notes

- Keep your [`.env`](ecommerce-chatbot/.env) file secure and never commit it to version control
- The [`.gitignore`](ecommerce-chatbot/.gitignore) file already excludes `.env` and `__pycache__/`
- Rotate your Groq API key periodically for security

## 📊 Data Management

### Adding New Products
Edit [`data/products.json`](ecommerce-chatbot/data/products.json):
```json
{
  "id": "PROD009",
  "name": "New Product",
  "category": "Category",
  "price": 99.99,
  "description": "Product description",
  "features": ["Feature 1", "Feature 2"],
  "in_stock": true,
  "rating": 4.5,
  "reviews_count": 100
}
```

### Adding New Orders
Edit [`data/orders.json`](ecommerce-chatbot/data/orders.json) with similar structure as existing orders.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational/demonstration purposes. Please ensure compliance with your organization's policies before commercial use.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure your Groq API key is valid and has sufficient credits
4. Check the console output for detailed error messages

---

**Happy Coding! 🎉**

For more information about the underlying technologies:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Groq API Documentation](https://console.groq.com/docs)