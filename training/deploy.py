from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

class ComponentRequest(BaseModel):
    component_code: str
    max_length: int = 200
    num_return_sequences: int = 1

# Load the fine-tuned model and tokenizer
model_name_or_path = "../fine_tuned/latest_50"  # Update this path
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

@app.post("/generate-test-cases/")
async def generate_test_cases(request: ComponentRequest):
    try:
        prompt = f"Generate a Jest and React Testing Library test case in TypeScript for the following React component:\n\n{request.component_code}\n\n// Test case:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=request.max_length,
            num_return_sequences=request.num_return_sequences,
            no_repeat_ngram_size=2,
            num_beams=5,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True  # Enable sampling
        )
        generated_test_cases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return {"generated_test_cases": generated_test_cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the test case generation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
