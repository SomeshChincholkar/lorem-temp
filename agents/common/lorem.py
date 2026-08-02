import os
import re

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()  # reads FA5_Project/.env (AWS creds, region, model id, etc.)

llm = ChatBedrockConverse(
    model_id=os.getenv("BEDROCK_MODEL_ID", "cohere.command-r-plus-v1:0"),
    region_name=os.getenv("AWS_REGION", "us-east-1"))

print(llm.invoke("HI"))