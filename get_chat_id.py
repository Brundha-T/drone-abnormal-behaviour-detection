import requests

TOKEN = "8472225035:AAHDvlDQ4lm7GuGN6hIE4POF3KQj7YZYjH0"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

print("\nContacting Telegram...")
response = requests.get(url).json()

if not response.get("ok"):
    print(f"Error from Telegram: {response}")
    print("\nPlease make sure you have sent a message (like 'Hello') to your bot on Telegram first!")
else:
    results = response.get("result", [])
    if not results:
        print("\nNo messages found! Please open Telegram, search for your bot, and send it a message like 'Hello' right now.")
    else:
        # Get the chat ID from the most recent message
        chat_id = results[-1]["message"]["chat"]["id"]
        print(f"\nSUCCESS! Your Chat ID is: {chat_id}")
        print(f"Paste this into line 25 of your app.py:")
        print(f'TELEGRAM_CHAT_ID = "{chat_id}"')
