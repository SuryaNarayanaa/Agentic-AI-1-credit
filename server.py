from fastapi import FastAPI, UploadFile, File, Form
import uvicorn

from agents import chat_agent

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat/")
def chat_endpoint(userQuery: str = Form(...), prompt_type: str = Form("cot"), url: str = Form(None), file: UploadFile = File(None)):
    file_content = None
    if file:
        file_content = file.file.read().decode('utf-8')
    return chat_agent(userQuery, prompt_type, url, file_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
