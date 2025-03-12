# i2_bridge

## Setup

## Run Locally
Install dependencies:
```bash
pip install -r requirements.txt
```

Start the server:
```bash
uvicorn app.main:app --reload
```

## API to analyze python repository
```
curl --location --request POST 'localhost:8000/api/v1/kg?github_url=https%3A%2F%2Fgithub.com%2Fpsf%2Frequests.git&branch=main'
```

## Deploy
```bash
docker build -t i2-bridge:1.0.0 .
helm install i2-bridge ./deploy/helm/i2-bridge
```

## Test
```bash
poetry run pytest

## API
curl --location 'localhost:8000/api/v1/kg?github_url=https://github.com/psf/requests.git'
