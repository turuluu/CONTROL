import requests
import io
from datetime import datetime

class Timed(io.StringIO):
    """Typed matching helper for Tee - see Tee below for refernce"""
    def now(self):
        return datetime.now().strftime('%H%M%S.%f')


class Tee(io.TextIOBase):
    """Save print outs in buffers

    Example:
        buffer_out = Timed()
        buffer_err = Timed()
        import sys
        tee_out = Tee(buffer_out, sys.stdout)
        tee_err = Tee(buffer_err, sys.stderr)

        # Use redirect_stdout to install it temporarily
        from contextlib import redirect_stdout, redirect_stderr
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("This will appear on-screen and saved in buffer")
            print('This is an error', file=sys.stderr)
            print("…and both are captured")

        lines = buffer_out.getvalue().splitlines() + buffer_err.getvalue().splitlines()
        lines.sort()
        for line in lines:
            print(line)

    """
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            if type(s) == Timed and data != '\n':
                s.write(f'{s.now()}| {data}')
            else:
                s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def download_file(url, save_path=None):
    """
    Downloads a binary file from a URL and saves it locally.

    Args:
        url (str): The URL of the file to download.
        save_path (str, optional): Local path to save the file. If not provided, filename is derived from URL.

    Raises:
        Exception: If the download fails.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Derive filename from URL if not provided
        if save_path is None:
            local_filename = url.split("/")[-1] or "downloaded_file"
        else:
            local_filename = save_path

        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

        return local_filename

    except Exception as e:
        print(f"Error downloading file: {e}")
        return None