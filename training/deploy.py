from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from fastapi import FastAPI, Request

app = FastAPI()

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2')
model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.post("/generate-test")
async def generate_test(request: Request):
    data = await request.json()
    component_code = data['component_code']
    generated = generator(f"Component:\n{component_code}\nTests:\n", max_length=512)
    return {"generated_tests": generated[0]['generated_text']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
