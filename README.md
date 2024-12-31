Steps to run the api:

0.Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
0.1. Create a .env file and add the API key
```bash
touch .env
```
```bash
echo "gemini_api_key = api_key" >> .env
```
1.Install the requirements
```bash
pip install -r requirements.txt
```
2.Run the app
```bash
flask run
```

API Endpoints:

1.Add context: /context (POST)
Request:
```json
{
    "context": "Context"
}
```

2.Get Response /response (GET)
Request:
```json
{
    "query": "Query"
}
```
