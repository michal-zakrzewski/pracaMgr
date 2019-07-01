from sys import platform
import shutil
import os.path
from os import path

if platform == "linux" or platform == "linux2":
    from IPython.core.display import display
    path = str("/content/pracaMgr")
# TODO: Add possibility to upload weights from local to Google Drive
# elif platform == "darwin":
#     path = str("./")
#     from google.colab import drive
#     drive.mount('/content/drive', force_remount=True)
# elif platform == "win32":
#     path = str("../")

# Create example file
with open("Example_file.txt", "w") as f:
    print("This is a test file", file=f)

try:
    shutil.copy(path + "/Example_file.txt", "/content/drive/My Drive/pracaMgr/Weights/Example_file.txt")
    os.path.exists("/Example_file.txt")
    try:
        os.path.exists("/content/drive/My Drive/pracaMgr/Weights/Example_file.txt")
        print("Following path exists, file was successfully exported")
        os.remove("/content/drive/My Drive/pracaMgr/Weights/Example_file.txt")
    except Exception as e:
        print("Checking the file out failed!")
        print(e)
except Exception as e:
    print('Saving was not possible, sorry')
    print(e)
finally:
    os.remove("Example_file.txt")
