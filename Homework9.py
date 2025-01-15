from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import logging

BOT_TOKEN = "7015015867:AAEI94RC9cy3QWNJiisvR-FXRniiYlp6-og"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        user_message = update.message.text
        print(f"Message received: {user_message}")
        response = process_message(user_message)
        print(f"Response from backend: {response}")
        await update.message.reply_text(response)
    else:
        print(f"Received non-text message: {update}")


def process_message(message):
    backend_url = "http://127.0.0.1:5000/llm"  # Make sure this is correct
    try:
        response = requests.post(backend_url, json={"message": message})
        model_response = response.json().get("response", "Sorry, I couldn't process that.")

        if model_response.lower().startswith(message.lower()):
            model_response = model_response[len(message):].strip()

        return model_response
    except Exception as e:
        return f"Error: {e}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler."""
    await update.message.reply_text("Hello! I'm your DSSS HW9 bot. How can I help you today? "
                                    "Do you want to know something about your favorite animal?")


def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    application.run_polling()




if __name__ == "__main__":
    main()
