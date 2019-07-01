from sys import platform
import shutil
import os

if platform == "linux" or platform == "linux2":
    from IPython.core.display import display
    path = str("/content/pracaMgr/input")
# TODO: Add possibility to upload weights from local to Google Drive
elif platform == "darwin":
    path = str("./input")
elif platform == "win32":
    path = str("../input")

# Create example file
with open("Example_file.txt", "a") as f:
    print("This is a test file", file=f)

try:
    shutil.move(path + "/Example_file.txt", "/content/drive/My Drive/pracaMgr/Weights/Example_file.txt")
except Exception as e:
    print('Saving was not possible, sorry')
    print(e)
finally:
    os.remove(path + "/Example_file.txt")
