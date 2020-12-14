import easyocr
import pathlib

files = pathlib.Path("example_data").glob("*.png")
reader = easyocr.Reader(['en'])

for file in files:
    print(reader.readtext(str(file)))
    print(file.name[:2])
    print("----------------------")