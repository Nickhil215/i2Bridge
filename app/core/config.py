from pydantic_settings import BaseSettings
import semver

class Settings(BaseSettings):
    # API
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # OpenTelemetry
    OTEL_SERVICE_NAME: str = "i2-bridge"
    OTEL_COLLECTOR_URL: str = "http://otel-collector:4317"

    ONTOLOGY_ID: str = "67d26d7163da105954cd5ac4"
    CONTENT_SERVICE_URL: str = "https://ig.aidtaas.com/mobius-content-service/v1.0/content/upload?filePath=python"
    ONTOLOGY_SERVICE_URL: str = "https://ig.aidtaas.com/pi-ontology-service/ontology/v1.0/patch?graphDb=NEO4J"

    @property
    def semver(self) -> semver.VersionInfo:
        return semver.VersionInfo.parse(self.VERSION)
    
    class Config:
        case_sensitive = True
        env_file = ".env"

