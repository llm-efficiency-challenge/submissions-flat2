from transformers import pipeline
import transformers

generator = pipeline(
    'text-generation',
    # model='EleutherAI/gpt-neo-125m',
    model='EleutherAI/gpt-neo-2.7B',
)
text = generator("EleutherAI has", do_sample=True, min_length=20)
print(text)