from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
save_path = "./models/flan-t5-base"

print("⏳ Downloading model... (this may take a few minutes)")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Model downloaded and saved to ./models/flan-t5-base")


#AutoTokenizer → Converts text → tokens (numbers)

#AutoModelForSeq2SeqLM → Loads the FLAN-T5 model that generates text