# run_task4.py
import subprocess
import sys

def install_requirements():
    print("📦 جاري تثبيت المتطلبات...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_embedding():
    print("🚀 جاري تشغيل Task 4...")
    subprocess.check_call([sys.executable, "embedding_generator.py"])

if __name__ == "__main__":
    install_requirements()
    run_embedding()
    