import json
from urllib import request, error

url = "http://localhost:8000/ask/"
payload = {"question": "When are refunds issued?"}

data = json.dumps(payload).encode("utf-8")
req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

try:
	with request.urlopen(req) as resp:
		status = resp.getcode()
		body = resp.read().decode("utf-8")
except error.HTTPError as e:
	status = e.code
	body = e.read().decode("utf-8")

print("Status Code:", status)
print("Response:", body)
