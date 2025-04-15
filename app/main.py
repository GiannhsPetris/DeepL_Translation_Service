import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Any, Dict, List, Union, Optional
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="EU Farmbook Translation Service",
              description="Standalone FastAPI microservice for the translation of documents and JSON files using DeepL API.")


class DocumentMetadata(BaseModel):
    id: str = Field(..., alias="@id")
    object_name: str
    object_hash: str
    object_extension: str
    object_size: int

    class Config:
        populate_by_name = True


class TranslateDocumentRequest(BaseModel):
    object_metadata: DocumentMetadata
    target_lang: str
    source_lang: Optional[str] = None


@app.post("/translate-document")
async def translate_document(request: TranslateDocumentRequest):
    import tempfile
    import os
    import deepl

    try:
        file_url = request.object_metadata.id
        if not file_url:
            raise HTTPException(status_code=400, detail="Missing file URL in object metadata")

        file_name = request.object_metadata.object_name
        file_ext = request.object_metadata.object_extension

        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            if response.status_code != 200:
                raise HTTPException(status_code=500,
                                    detail=f"Failed to download file: HTTP {response.status_code}")
            contents = response.content

        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="DeepL API key not configured")

        deepl_client = deepl.Translator(api_key)

        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_input:
            temp_input.write(contents)
            input_path = temp_input.name

        output_path = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False).name


        deepl_client.translate_document_from_filepath(
            input_path,
            output_path,
            target_lang=request.target_lang,
            source_lang=request.source_lang
        )

        with open(output_path, "rb") as output_file:
            translated_content = output_file.read()

        output_filename = f"translated_{file_name}"

        os.unlink(input_path)
        os.unlink(output_path)

        return StreamingResponse(
            content=iter([translated_content]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )

    except deepl.DeepLException as e:
        raise HTTPException(status_code=500, detail=f"DeepL API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating document: {str(e)}")


@app.post("/translate-json")
async def translate_json(
    file: UploadFile = File(...),
    target_lang: str = Form(...),
    source_lang: str = Form(None)
):
    import deepl
    import json

    try:
        contents = await file.read()
        try:
            data = json.loads(contents.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")

        api_key = os.getenv("DEEPL_API_KEY")
        print(f"apikey is {api_key}")
        if not api_key:
            raise HTTPException(status_code=500, detail="DeepL API key not configured")

        deepl_client = deepl.Translator(api_key)

        # Process the JSON to translate only values
        translated_data = await translate_json_values(
            data,
            deepl_client,
            target_lang,
            source_lang
        )

        return translated_data

    except deepl.DeepLException as e:
        raise HTTPException(status_code=500, detail=f"DeepL API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating JSON: {str(e)}")


@app.get("/deepl-usage")
async def get_deepl_usage():
    import deepl
    try:
        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="DeepL API key not configured")

        deepl_client = deepl.Translator(api_key)
        return deepl_client.get_usage()

    except deepl.DeepLException as e:
        raise HTTPException(status_code=500, detail=f"DeepL API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking DeepL usage: {str(e)}")


async def translate_json_values(
        data: Union[Dict, List, str, int, float, bool, None],
        translator: Any,
        target_lang: str,
        source_lang: str = None
) -> Union[Dict, List, str, int, float, bool, None]:
    """Recursively translate values in JSON while preserving keys and structure"""

    # Handle dictionary
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = await translate_json_values(value, translator, target_lang, source_lang)
        return result

    # Handle list
    elif isinstance(data, list):
        return [await translate_json_values(item, translator, target_lang, source_lang) for item in data]

    # Translate string values
    elif isinstance(data, str) and data.strip():
        translation = translator.translate_text(
            data,
            target_lang=target_lang,
            source_lang=source_lang,
        )
        return str(translation)

    # Return non-translatable values as is
    else:
        return data