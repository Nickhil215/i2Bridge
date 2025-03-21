import json
import logging
import uuid
from io import BytesIO

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from app import logger, settings
from app.core.exceptions import ApiException
from app.services.package_analyzer_factory import PackageAnalyzerFactory


def analyze_package(package_path: str, branch: str, **kwargs):

    try:
        with PackageAnalyzerFactory.create_analyzer(package_path, branch, **kwargs) as analyzer:

            analyzer.analyze_package()
            # txt_output = analyzer.generate_report()
            ttl_output = analyzer.convert_analyzer_to_knowledge_graph()
            function_list = analyzer.get_functions_list()
            return ttl_output, function_list

    except Exception as e:
        error_msg = f"Error analyzing package: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def upload_to_content_service(content, bearer_token):
    global response
    file_name = f"ttl_output_{uuid.uuid1().hex}.ttl"

    url = settings.CONTENT_SERVICE_URL

    # Headers including Bearer token
    headers = {
        'Authorization': bearer_token
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
    except requests.exceptions.HTTPError as e:
        logging.error(f"API request error: {e}")
        error = None
        if response is not None:
            error = response.text
        raise ApiException("Failed to make API call", 500, error)
    except ValueError as e:
        logging.error(f"JSON parsing error: {e}")
        raise ApiException("Object mapping failure", 500)


def import_kg(cdn_url, bearer_token):

    global response
    url = settings.ONTOLOGY_SERVICE_URL
    headers = {
        'Authorization': bearer_token,
        'Content-Type': 'application/json',
    }

    data = {
        "ontologyId": settings.ONTOLOGY_ID,
        "universes": [
            "623ebf349279af04cdd709dc"
        ],
        "url": f'https://cdn.aidtaas.com{cdn_url}',
        "ontologyName": "API",
        "semanticStructures": "DATA",
        "fileType": "Turtle",
        "tenantID": "2cf76e5f-26ad-4f2c-bccc-f4bc1e7bfb64"
    }

    try:
        response = requests.patch(url, headers=headers, json=data)
        response.raise_for_status()
        logging.info(f"------- Response body:  {response.text} --------")
    except requests.exceptions.HTTPError as e:
        logging.error(f"API request error: {e}")
        error = None
        if response is not None:
            error = response.text
        raise ApiException("Failed to make API call", 500, error)
    except ValueError as e:
        logging.error(f"JSON parsing error: {e}")
        raise ApiException("Object mapping failure", 500)


def analyze(package_path, branch, request):
    ttl_output, function_list = analyze_package(package_path, branch)

    # with open("/home/gaian/Desktop/python/i2_bridge/sample/output.ttl", "w", encoding="utf-8") as f:
    #     f.write(ttl_output)

    # with open("/home/gaian/Desktop/python/i2_bridge/sample/output.txt", 'w', encoding='utf-8') as f:
    #     f.write(txt_output)

    bearer_token = request.headers.get("Authorization")
    cdn_url = upload_to_content_service(ttl_output, bearer_token)
    logging.info(f"------- cdn_url:  {cdn_url} --------")
    # import_kg(cdn_url, bearer_token)

    function_list["cdn_url"] = cdn_url

    return function_list

