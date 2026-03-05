#credentials = "C:\Users\Elevate\AppData\Roaming\gcloud\application_default_credentials.json"
import vertexai
from vertexai import rag

# Vertex AI settings
PROJECT_ID = "knowledge-base-agent-485813"
LOCATION = "europe-west1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Import agent after initialization is complete
from . import agent