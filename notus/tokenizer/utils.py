from .all_file_ext import file_extensions

def get_file_ext_as_token(filename):
    ext = filename.split('.')[-1]
    try:
        return file_extensions["."+ext]
    except KeyError:
        assert "unknow extension"