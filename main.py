import os
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

## Load env variables
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") #type: ignore

class ChatMessages(BaseModel):
    messages : str  = Field(description="The messages in the conversation")
    title: str = Field(description="Title of the paragraph")

model = init_chat_model(model="gemini-1.5-flash", model_provider="google_genai")
structured_llm = model.with_structured_output(ChatMessages)
output = structured_llm.invoke(
    "Tell me something about Artifivial General initelligence"
)
print(output.messages)





