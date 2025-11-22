import time
import os
import requests
from simplegmail import Gmail
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyPDF2 import PdfReader

# CONFIGURATION
API_URL = "https://demo.indxai.tech/train"  # CHANGE THIS to your Render URL
WATCH_FOLDER = "indxai_knowledge_drop"  # Create this folder on your desktop

print("╔══════════════════════════════════════════╗")
print("║   indxai REAL-TIME DATA BRIDGE v1.0      ║")
print("╚══════════════════════════════════════════╝")


# ==========================================
# 1. GMAIL CONNECTOR (Real OAuth)
# ==========================================
class GmailBridge:
    def __init__(self):
        print("📧 Initializing Gmail Bridge...")
        # This looks for 'client_secret.json' and opens browser to login
        try:
            self.gmail = Gmail()
            print("✅ Gmail Connected Successfully!")
        except Exception as e:
            print(f"⚠️ Gmail Auth Failed: {e}")
            print("   (Did you download client_secret.json from Google Cloud?)")
            self.gmail = None

    def fetch_unread(self):
        if not self.gmail:
            return []

        messages = self.gmail.get_unread_inbox()
        data_batch = []

        for message in messages:
            # Mark as read so we don't re-learn it 100 times
            # message.mark_as_read()

            text = f"Email from {message.sender}: {message.snippet}"
            print(f"   📩 Found email: {message.subject[:30]}...")
            data_batch.append(text)

        return data_batch


# ==========================================
# 2. FILE SYSTEM WATCHER (The "Magic Folder")
# ==========================================
class MagicFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        print(f"📄 New file detected: {filepath}")
        time.sleep(1)  # Wait for write to finish

        content = ""
        try:
            if filepath.endswith(".pdf"):
                reader = PdfReader(filepath)
                content = " ".join([page.extract_text() for page in reader.pages])
            elif filepath.endswith(".txt") or filepath.endswith(".md"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

            if content:
                print(f"   extracting {len(content)} chars...")
                push_to_cloud(content, source=f"File: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")


# ==========================================
# 3. CLOUD UPLOAD UTILITY
# ==========================================
def push_to_cloud(text, source="Auto"):
    """Sends real data to your Render AI"""
    if len(text) < 5:
        return

    try:
        # Add source tag so AI knows where it came from
        tagged_text = f"[{source}]: {text}"

        response = requests.post(API_URL, data={"text_data": tagged_text})
        if response.status_code == 200:
            print(f"   🚀 UPLOADED to AI Brain. Knowledge Updated.")
        else:
            print(f"   ❌ Server Error: {response.text}")
    except Exception as e:
        print(f"   ⚠️ Connection Failed (Check URL): {e}")


# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Setup Magic Folder
    if not os.path.exists(WATCH_FOLDER):
        os.makedirs(WATCH_FOLDER)
        print(f"📁 Created folder '{WATCH_FOLDER}'. Drop PDFs here!")

    # 2. Start File Watcher
    observer = Observer()
    observer.schedule(MagicFolderHandler(), path=WATCH_FOLDER, recursive=False)
    observer.start()
    print(f"👀 Watching '{WATCH_FOLDER}' for new documents...")

    # 3. Setup Gmail
    gmail_bot = GmailBridge()

    try:
        while True:
            # Poll Gmail every 30 seconds
            if gmail_bot.gmail:
                emails = gmail_bot.fetch_unread()
                for email_text in emails:
                    push_to_cloud(email_text, source="Gmail")

            time.sleep(30)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
