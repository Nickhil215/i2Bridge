import json
import logging
import sys
import uuid
from io import BytesIO

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from app.core.exceptions import ApiException
from app import logger
from app.services.package_analyzer_factory import PackageAnalyzerFactory


def analyze_package(package_path: str, **kwargs) -> str:

    try:
        with PackageAnalyzerFactory.create_analyzer(package_path, **kwargs) as analyzer:

            analyzer.analyze_package()
            txt_output = analyzer.generate_report()
            ttl_output = analyzer.convert_analyzer_to_knowledge_graph()
            function_list = analyzer.get_functions_list()
            return ttl_output, txt_output, function_list

    except Exception as e:
        error_msg = f"Error analyzing package: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def upload_to_content_service(content, bearer_token):
    file_name = f"ttl_output_{uuid.uuid1().hex}.ttl"

    url = "https://ig.aidtaas.com/mobius-content-service/v1.0/content/upload?filePath=python"

    # Headers including Bearer token
    headers = {
        'Authorization': f'Bearer {bearer_token}',
    }

    # Create the file part of the multipart body
    multipart_data = MultipartEncoder(
        fields={
            'file': (file_name, BytesIO(content.encode('utf-8')), 'application/octet-stream'),
        }
    )

    # Update the headers with the correct multipart content type
    headers['Content-Type'] = multipart_data.content_type

    try:
        response = requests.post(url, data=multipart_data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses
        logging.info(f"------- Response body:  {response.text} --------")
        return json.loads(json.dumps(response.json())).get("cdnUrl")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request error: {e}")
        raise ApiException("Failed to make API call")
    except ValueError as e:
        logging.error(f"JSON parsing error: {e}")
        raise ApiException("Object mapping failure")


def import_ontology(cdn_url, bearer_token):

    url = 'https://ig.aidtaas.com/pi-ontology-service/ontology/v1.0/patch?graphDb=NEO4J'
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }

    data = {
        "ontologyId": "67c7f94236726044b0df97a8",
        "universes": [
            "623ebf349279af04cdd709dc"
        ],
        "url": cdn_url,
        "ontologyName": "API",
        "semanticStructures": "DATA",
        "fileType": "Turtle",
        "tenantID": "2cf76e5f-26ad-4f2c-bccc-f4bc1e7bfb64"
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logging.info(f"------- Response body:  {response.text} --------")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request error: {e}")
        raise ApiException("Failed to make API call")
    except ValueError as e:
        logging.error(f"JSON parsing error: {e}")
        raise ApiException("Object mapping failure")


# def import_ontology(cdn_url, bearer_token):
#
#     url = 'https://ig.aidtaas.com/pi-ontology-service/ontology/v1.0/create'
#     headers = {
#         'Authorization': f'Bearer {bearer_token}',
#         'Content-Type': 'application/json',
#     }
#
#     data = {
#         "ontologyUrl": cdn_url,
#         "ontologyName": "API",
#         "description": "This is a sample ontology",
#         "fileType": "TURTLE",
#         "semanticStructures": "DATA",
#         "defaultPromptTemplates": "USER",
#         "userPromptForKgCreation": "consider all the data",
#         "draft": False,
#         "dataBaseType": "NEO4J",
#         "universes": [
#             "66aa30f77daee22fb1f1d214"
#         ],
#         "isActive": True,
#         "tenantID": "12345",
#         "visibility": "PUBLIC",
#         "dataReadAccess": "PUBLIC",
#         "dataWriteAccess": "PUBLIC",
#         "metadataReadAccess": "PUBLIC",
#         "metadataWriteAccess": "PUBLIC",
#         "execute": "PUBLIC"
#     }
#
#     try:
#         response = requests.post(url, headers=headers, json=data)
#         response.raise_for_status()
#         logging.info(f"------- Response body:  {response.text} --------")
#     except requests.exceptions.RequestException as e:
#         logging.error(f"API request error: {e}")
#         raise ApiException("Failed to make API call")
#     except ValueError as e:
#         logging.error(f"JSON parsing error: {e}")
#         raise ApiException("Object mapping failure")

def analyze(package_path, request):
    try:
        ttl_output, txt_output, function_list = analyze_package(package_path)

        # with open("/home/gaian/Desktop/python/i2_bridge/sample/output.ttl", "w", encoding="utf-8") as f:
        #     f.write(ttl_output)
        #
        # with open("/home/gaian/Desktop/python/i2_bridge/sample/output.txt", 'w', encoding='utf-8') as f:
        #     f.write(txt_output)

        # bearer_token = request.headers.get("Authorization")
        # cdn_url = upload_to_content_service(ttl_output, bearer_token)
        # logging.info(f"------- cdn_url:  {cdn_url} --------")
        # import_ontology(cdn_url, bearer_token)

        return function_list

    except Exception as e:
        logger.error(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

