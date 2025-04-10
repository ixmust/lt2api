from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
import json
import uuid
import time
import logging
import random
from typing import List, Dict, Any, AsyncGenerator
from functools import wraps
from pydantic import BaseModel, Field

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAIMessage(BaseModel):
    role: str
    content: str | Dict[str, Any] | List[Any] = Field(default="")  # 支持多种类型


class OpenAIRequest(BaseModel):
    messages: List[OpenAIMessage]
    stream: bool = Field(default=False)


class WoHistoryItem(BaseModel):
    query: str
    rewriteQuery: str
    uploadFileUrl: str
    response: str
    reasoningContent: str
    state: str
    key: str


class WoRequest(BaseModel):
    modelId: int
    input: str
    history: List[WoHistoryItem]


def validate_messages(f):
    """Message format validation decorator"""

    @wraps(f)
    async def decorated_function(*args, **kwargs):
        request: Request = kwargs.get('request')  # type: ignore
        if not request:
            raise HTTPException(status_code=500, detail="Missing request object")
        body = await request.json()
        messages = body.get('messages', [])

        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                raise HTTPException(status_code=400, detail="Invalid message format")
        return await f(*args, **kwargs)

    return decorated_function


def convert_history(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert OpenAI format messages to WoCloud format"""
    history = []

    for i in range(len(messages) - 1):
        try:
            if messages[i]['role'] == 'user' and i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                query = messages[i]['content']
                response = messages[i + 1]['content']
                
                # Convert JSON content to string if necessary
                if not isinstance(query, str):
                    query = json.dumps(query)
                if not isinstance(response, str):
                    response = json.dumps(response)
                
                query = query.strip()
                response = response.strip()

                history.append({
                    "query": query,
                    "rewriteQuery": query,
                    "uploadFileUrl": "",
                    "response": response,
                    "reasoningContent": "",
                    "state": "finish",
                    "key": str(random.random())
                })
        except (KeyError, IndexError) as e:
            logger.warning(f"Error processing message: {str(e)}")
            continue

    logger.debug(f"Converted history: {json.dumps(history, ensure_ascii=False)}")
    return history


async def handle_wo_error(response: httpx.Response) -> Dict[str, str]:
    """Unified handling of WoCloud error responses"""
    try:
        if response.headers.get('Content-Type', '').startswith('text/event-stream'):
            async for line in response.aiter_lines():
                if line.startswith('data:'):
                    try:
                        error_data = json.loads(line[5:].strip())
                        return {
                            "code": error_data.get('code'),
                            "message": error_data.get('message', 'Unknown error')
                        }
                    except json.JSONDecodeError:
                        return {
                            "code": "PARSE_ERROR",
                            "message": f"Invalid JSON in error response: {line[5:100]}"
                        }
        else:
            try:
                error_data = response.json()
                return {
                    "code": error_data.get('code'),
                    "message": error_data.get('message', error_data.get('response', 'Unknown error'))
                }
            except json.JSONDecodeError:
                return {
                    "code": "PARSE_ERROR",
                    "message": f"Invalid JSON in error response: {response.text[:100]}"
                }
    except Exception as e:
        return {
            "code": "PARSE_ERROR",
            "message": f"Failed to parse error response: {str(e)}"
        }


def create_chunk(content: str, response_id: str, created_time: int, finish_reason: str = None,
                 is_start: bool = False, role: str = None) -> str:
    """Creates a single chunk in the OpenAI streaming format."""
    chunk_data = {
        'id': response_id,
        'object': 'chat.completion.chunk',
        'created': created_time,
        'model': 'DeepSeek-R1',
        'choices': [{
            'index': 0,
            'delta': {},
            'finish_reason': finish_reason
        }]
    }
    if is_start and role:
         chunk_data["choices"][0]["delta"]["role"] = role
    elif content:
        chunk_data['choices'][0]['delta']['content'] = content


    return f"data: {json.dumps(chunk_data)}\n\n"


@app.post('/v1/chat/completions')
@validate_messages
async def chat_completions(request: Request, authorization: str = Header(...)):
    if not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    access_token = authorization[7:]
    openai_request = await request.json()
    stream_mode = openai_request.get('stream', False)

    # Extract last user message
    user_message = next(
        (msg['content'] for msg in reversed(openai_request['messages'])
         if msg['role'] == 'user' and msg.get('content')),
        None
    )
    if not user_message:
        raise HTTPException(status_code=400, detail="No valid user message found")
    
    # Convert JSON content to string if necessary
    if not isinstance(user_message, str):
        user_message = json.dumps(user_message)
    
    # Build WoCloud request
    wo_headers = {
        'content-type': 'application/json',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'connection': 'keep-alive',
        'dnt': '1',
        'origin': 'https://panservice.mail.wo.cn',
        'referer': f'https://panservice.mail.wo.cn/h5/wocloud_ai/?token={access_token}&modelType=1&platform=yunpanWeb&clientId=1001000021',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1 Edg/134.0.0.0',
        'x-yp-access-token': access_token,
        'x-yp-app-version': '',  # 注意保持空值
        'x-yp-client-id': '1001000021',  # 原值为1001000035
        'accept': 'text/event-stream' if stream_mode else 'application/json'
    }

    # Convert history
    history = convert_history(openai_request['messages'][:-1])  # Exclude last user message
    wo_data = {
        "modelId": 1,
        "input": user_message,
        "history": history
    }
    logger.debug(f"Forwarding request to WoCloud: {json.dumps(wo_data, ensure_ascii=False)}")

    async def stream_response() -> AsyncGenerator[bytes, None]:
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        full_content = ""

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                        'POST',
                        'https://panservice.mail.wo.cn/wohome/ai/assistant/query',
                        headers=wo_headers,
                        json=wo_data,
                        timeout=30
                ) as response:

                    logger.debug(f"WoCloud stream response status: {response.status_code}")

                    if response.status_code != 200:
                        error_info = await handle_wo_error(response)
                        logger.error(f"WoCloud API error: {error_info}")
                        yield f"data: {json.dumps({'error': error_info})}\n\n".encode('utf-8')
                        return

                    # Send initial chunk with role
                    yield create_chunk("", response_id, created_time, is_start=True, role="assistant").encode("utf-8")

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                if line.startswith('data:'):
                                    data = json.loads(line[5:])

                                    if data.get('code') and data['code'] != 0 and data['code'] != '0':
                                        logger.error(f"WoCloud stream error: {data}")
                                        yield f"data: {json.dumps({'error': {'code': data['code'], 'message': data.get('message', 'Unknown error')}})}\n\n".encode(
                                            'utf-8')
                                        return

                                    content = data.get('response', '')
                                    reasoning = data.get('reasoningContent', '')

                                    if reasoning:
                                        if not full_content:
                                             yield create_chunk("<think>\n", response_id, created_time).encode("utf-8")

                                        full_content += reasoning
                                        yield create_chunk(reasoning, response_id, created_time).encode("utf-8")


                                    if content:
                                        if full_content:  # If there's thinking content, end thinking first
                                            yield create_chunk("\n</think>\n\n", response_id, created_time).encode("utf-8")
                                            full_content = ""
                                        yield create_chunk(content, response_id, created_time).encode("utf-8")


                                    if data.get("finish") == 1:
                                        break

                            except Exception as e:
                                logger.error(f"Stream parsing error: {str(e)}")
                                yield f"data: {json.dumps({'error': {'code': 'PARSE_ERROR', 'message': str(e)}})}\n\n".encode(
                                    'utf-8')
            except httpx.RequestError as e:
                logger.error(f"Request failed: {str(e)}")
                yield f"data: {json.dumps({'error': {'code': 'CONNECTION_ERROR', 'message': str(e)}})}\n\n".encode(
                    'utf-8')

            yield create_chunk("", response_id, created_time, finish_reason="stop").encode("utf-8")
            yield "data: [DONE]\n\n".encode('utf-8')

    if stream_mode:
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    'https://panservice.mail.wo.cn/wohome/ai/assistant/query',
                    headers=wo_headers,
                    json=wo_data,
                    timeout=30
                )
                logger.debug(f"WoCloud response status: {response.status_code}")
                logger.debug(f"WoCloud response headers: {response.headers}")
                if response.status_code != 200:
                    error_info = await handle_wo_error(response)
                    logger.error(f"WoCloud API error: {error_info}")
                    raise HTTPException(status_code=502, detail=error_info)

                content = ""

                if response.headers.get('Content-Type', '').startswith('application/json'):
                    try:
                        response_data = response.json()
                        if response_data.get('code') and response_data['code'] != 0 and response_data['code'] != '0':
                            logger.error(f"WoCloud API error: {response_data}")
                            raise HTTPException(status_code=502, detail={
                                "code": response_data['code'],
                                "message": response_data.get("message", "Unknown error")
                            })

                        content = response_data.get('response', '')
                        reasoning = response_data.get('reasoningContent', '')

                        final_content = ""
                        if reasoning:
                            final_content += f"<think>\n{reasoning}\n</think>\n\n"
                        final_content += content
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON Response:{response.text[:200]}")
                        raise HTTPException(status_code=502,
                                            detail={"code": "PARSE_ERROR", "message": "Failed to parse JSON response"})
                else:
                    response_data = {"response": "", "reasoningContent": ""}
                    try:
                        for line in response.text.split('\n'):
                            if line.startswith('data:'):
                                try:
                                    data = json.loads(line[5:])
                                    if data.get('code') and data['code'] != 0 and data['code'] != '0':
                                        logger.error(f"WoCloud API error in stream:{data}")
                                        raise HTTPException(status_code=502, detail={
                                            "code": data['code'],
                                            "message": data.get('message', 'Unknown error')
                                        })
                                    response_data['response'] += data.get('response', '')
                                    response_data['reasoningContent'] += data.get('reasoningContent', '')
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid JSON in stream line: {line[:100]}")
                        final_content = ''
                        if response_data['reasoningContent']:
                            final_content += f"<think>\n{response_data['reasoningContent']}\n</think>\n\n"
                        final_content += response_data['response']

                    except Exception as e:
                        logger.error(f"Error parsing stream response:{str(e)}")
                        raise HTTPException(status_code=502, detail={
                            "code": "PARSE_ERROR",
                            "message": f"Failed to parse stream response: {str(e)}"
                        })

                if not final_content:
                    logger.warning("Empty content in response")
                    final_content = "抱歉，没有收到有效的回复。"

                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "DeepSeek-R1",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(user_message),
                        "completion_tokens": len(final_content),
                        "total_tokens": len(user_message) + len(final_content)
                    }
                }


            except httpx.RequestError as e:
                logger.error(f"Request failed: {str(e)}")
                raise HTTPException(status_code=502, detail={
                    "code": "NETWORK_ERROR",
                    "message": str(e)
                })
            except Exception as e:
                logger.exception("Unexpected error")
                raise HTTPException(status_code=500, detail={
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                })


@app.get('/v1/models')
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "DeepSeek-R1",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ChinaUnicom",
            "capabilities": ["chat", "completions"]
        }]
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)