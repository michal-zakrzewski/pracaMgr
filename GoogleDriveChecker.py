from sys import platform
import shutil
import os.path
from os import path

if platform == "linux" or platform == "linux2":
    from IPython.core.display import display
    path = str("/content/pracaMgr")
elif platform == "darwin":
    path = str("./")
    # from google.colab import drive
    # drive.mount('/content/drive', force_remount=True)
elif platform == "win32":
    path = str("../")

"""
Skrypt sprawdzajacy poprawnosc polaczenia do dysku Google
W przypadku bledow cala operacja treningu/testu na Google Colab zostanie zatrzymana
Test opera sie na stworzeniu losowego pliku, eksporcie go na dysk Google, a nastepnie proba jego usuniecia
"""

working_directory = os.getcwd()
absolute_path = working_directory + '/Example_file.txt'

# Create example file
with open(absolute_path, "w") as f:
    print("This is a test file", file=f)

# Check if the file was created:
try:
    os.path.exists(absolute_path)
except OSError as e:
    print("File is not created correctly")
    raise

try:
    # Sprobuj skopiowac plik na dysk Google
    shutil.copy(absolute_path, "/content/drive/My Drive/pracaMgr/Weights/Example_file.txt")
    try:
        os.path.exists("/content/drive/My Drive/pracaMgr/Weights/Example_file.txt")
        print("Following path exists, file was successfully exported")
        os.remove("/content/drive/My Drive/pracaMgr/Weights/Example_file.txt")
    except Exception as e:
        print("File is not on Google Drive!")
        print(e)
        raise
except OSError as e:
    print("Problems with a path - it does not exist")
    print(absolute_path)
    raise
except Exception as e:
    print('Saving was not possible, sorry')
    print(e)
    raise
finally:
    os.remove("Example_file.txt")
