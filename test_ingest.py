import importlib
import os

# --- Dynamic import of requests ---
try:
    requests = importlib.import_module('requests')
    USING_REAL_REQUESTS = True
except Exception:
    import uuid
    import mimetypes
    import urllib.request
    import urllib.error

    class _Response:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

    class _RequestsFallback:
        def post(self, url, files=None):
            boundary = '----------{}'.format(uuid.uuid4().hex)
            body_parts = []
            if files:
                # files expected as dict: {'fieldname': file-like}
                for field, fp in files.items():
                    filename = getattr(fp, "name", "file")
                    content = fp.read()
                    content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                    body_parts.append(b'--' + boundary.encode('utf-8'))
                    body_parts.append(
                        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode('utf-8')
                    )
                    body_parts.append(f'Content-Type: {content_type}'.encode('utf-8'))
                    body_parts.append(b'')
                    body_parts.append(content)
            body_parts.append(b'--' + boundary.encode('utf-8') + b'--')
            body = b'\r\n'.join(body_parts) + b'\r\n'

            req = urllib.request.Request(url, data=body, method='POST')
            req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
            req.add_header('Content-Length', str(len(body)))
            try:
                with urllib.request.urlopen(req) as resp:
                    return _Response(resp.getcode(), resp.read().decode('utf-8', errors='replace'))
            except urllib.error.HTTPError as e:
                return _Response(e.code, e.read().decode('utf-8', errors='replace'))

    requests = _RequestsFallback()
    USING_REAL_REQUESTS = False

# --- Your endpoint ---
url = "http://localhost:8000/ingest/"

# --- Your PDF file list ---
pdf_files = [
    "sample.pdf",
    "sample_1.pdf",
    "sample_2.pdf",
    "sample_3.pdf",
    "sample_4.pdf",
    "sample_5.pdf",
    "sample_6.pdf",
    "sample_7.pdf",
    "sample_8.pdf",
    "sample_9.pdf",
    "sample_10.pdf",
]

# --- Build payload ---
if USING_REAL_REQUESTS:
    # real requests → list of tuples
    files = [('file', (os.path.basename(pdf), open(pdf, 'rb'))) for pdf in pdf_files]
    try:
        response = requests.post(url, files=files)
    finally:
        for _, file_tuple in files:
            file_tuple[1].close()
else:
    # fallback → dict (but we’ll send ONE file at a time)
    for pdf in pdf_files:
        with open(pdf, 'rb') as f:
            response = requests.post(url, files={'file': f})
            print(f"Uploaded {pdf} → Status Code: {response.status_code}")
            print("Response:", response.text)
    # exit after loop because we already posted all
    exit()

# --- Print if using real requests ---
print("Status Code:", response.status_code)
print("Response:", response.text)
