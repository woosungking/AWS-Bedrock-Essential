import boto3

def chunk_handler(chunk):
    print(chunk, end='')


def get_streaming_response(prompt, streaming_callback):
    
    session = boto3.Session()
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2')
    message = {
        "role": "user",
        "content": [ { "text": prompt } ]
    }
    
    response = bedrock.converse_stream(
        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0.0
        }
    )
    
    stream = response.get('stream')
    for event in stream:
        if "contentBlockDelta" in event:
            streaming_callback(event['contentBlockDelta']['delta']['text'])

prompt = "Tell me a story about two puppies and two kittens who became best friends:"
                
get_streaming_response(prompt, chunk_handler)
print("\n")




