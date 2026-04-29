
import boto3
def get_text_response(input_content):

    session = boto3.Session()
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2')
    
    message = {
        "role": "user",
        "content": [ { "text": input_content } ]
    }
    
    response = bedrock.converse(
        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0,
            "topP": 0.9,
            "stopSequences": []
        },
    )
    
    return response['output']['message']['content'][0]['text']
    

