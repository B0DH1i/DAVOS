import subprocess
import os
import sys

# Proje ana dizinini belirle (bu script'in bulunduğu dizin)
project_root = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(project_root) # gemini dizini

# Hedef sanal ortamdaki Python yorumlayıcısının yolu
venv_python_executable = os.path.join(workspace_root, ".venv-tf210-py310-gpu", "Scripts", "python.exe")

# Video dosyasının yolu (proje kök dizinine göre)
video_file_relative_path = os.path.join("data", "22597-328624850_small.mp4")
video_file_absolute_path = os.path.join(project_root, video_file_relative_path)

# Spesifik model klasör adları
fer_model_folder = "fer_vgg16_transfer_20250602-095125" # En son VGG16 FER modeli
ser_model_folder = "ser_model_simple_dense_whisper_whisper_20250602-230346" # En son SER (config ile aynı)

# Çalıştırılacak komut
command = [
    venv_python_executable,
    "-m", "src.main_live_controller",
    "--video_file", video_file_absolute_path,
    "--fer_model", fer_model_folder,
    "--ser_model", ser_model_folder
]

print(f"Çalıştırılacak Komut: {' '.join(command)}")

# Komutu proje ana dizininde çalıştır
try:
    # subprocess.run(command, check=True, cwd=project_root) # stderr'i de görmek için capture_output
    completed_process = subprocess.run(command, cwd=project_root, capture_output=True, text=True, check=False)
    
    print("\n----- Script Çıktısı (stdout) -----")
    print(completed_process.stdout)
    
    if completed_process.stderr:
        print("\n----- Script Hata Çıktısı (stderr) -----")
        print(completed_process.stderr)
        
    if completed_process.returncode != 0:
        print(f"\nScript {completed_process.returncode} hata koduyla sonlandı.")
    else:
        print("\nScript başarıyla tamamlandı.")

except FileNotFoundError:
    print(f"Hata: Sanal ortam Python yorumlayıcısı bulunamadı: {venv_python_executable}")
    print("Lütfen '.venv-tf210-py310-gpu' sanal ortamının 'gemini' klasörü altında olduğundan emin olun.")
except subprocess.CalledProcessError as e:
    print(f"Script çalıştırılırken hata oluştu (CalledProcessError): {e}")
    print("Stdout:")
    print(e.stdout)
    print("Stderr:")
    print(e.stderr)
except Exception as e:
    print(f"Script çalıştırılırken beklenmedik bir hata oluştu: {e}")

sys.exit(0) # Her zaman 0 ile çıkalım, terminal çıktısını Cursor'a bildirmek için. 