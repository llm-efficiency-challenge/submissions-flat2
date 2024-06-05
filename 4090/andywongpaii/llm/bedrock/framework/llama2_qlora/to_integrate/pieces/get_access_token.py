import getpass
import locale; locale.getpreferredencoding = lambda: "UTF-8"
import os

os.environ["HUGGING_FACE_HUB_TOKEN"] = getpass.getpass("Token:")
assert os.environ["HUGGING_FACE_HUB_TOKEN"]