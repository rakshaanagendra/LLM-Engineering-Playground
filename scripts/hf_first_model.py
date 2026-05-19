from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The future of artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50)

result = tokenizer.decode(outputs[0])

print(result)