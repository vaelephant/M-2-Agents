import http.client
import json

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({
  "q": "什么是五一劳动节"
})
headers = {
  #'X-API-KEY': '0b734c20f39b5aefc4b5d00601f1e967bd984a0c',
  'Content-Type': 'application/json'
}
conn.request("POST", "/search", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))