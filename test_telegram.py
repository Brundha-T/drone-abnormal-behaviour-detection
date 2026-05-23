import requests
import os

# ------------------------------------------------------------------ CONFIG
# These should match EXACTLY what you have in app.py
TOKEN = "8779510165:AAEGjJ6sEyH7GBXp07txllI5pruKIlkQL7Q"
CHAT_ID = "5348812660"

print("="*60)
print("🚀 TELEGRAM BOT DIAGNOSTIC TOOL")
print("="*60)
print(f"BOT TOKEN : {TOKEN[:10]}...{TOKEN[-5:]}")
print(f"CHAT ID   : {CHAT_ID}")
print("-" * 60)

def test_connection():
    url = f"https://api.telegram.org/bot{TOKEN}/getMe"
    try:
        print("[1] Testing Bot Connection...")
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("ok"):
            print(f"    ✅ SUCCESS: Connected to bot @{data['result']['username']}")
            return True
        else:
            print(f"    ❌ ERROR: {data.get('description')}")
            return False
    except Exception as e:
        print(f"    ❌ CONNECTION FAILED: {str(e)}")
        return False

def send_test_message():
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🛠️ *Diagnostic Test*: If you see this, your Bot Token and Chat ID are working correctly!",
        "parse_mode": "Markdown"
    }
    try:
        print("[2] Sending Test Message...")
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        if data.get("ok"):
            print("    ✅ SUCCESS: Message sent to Telegram!")
            return True
        else:
            print(f"    ❌ ERROR: {data.get('description')}")
            print("    👉 Check if you have started the bot (sent it a message first).")
            return False
    except Exception as e:
        print(f"    ❌ FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    if test_connection():
        send_test_message()
    print("="*60)
    print("If both tests PASSED but app.py still doesn't send alerts,")
    print("it might be a firewall or antivirus blocking the video upload.")
    print("="*60)
