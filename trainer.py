import os
from dotenv import load_dotenv
import cohere

load_dotenv()

co = cohere.Client(os.getenv('TEST_TOKEN'))

from cohere.responses.classify import Example

examples=[
  Example("I love you!", "positive"), 
  Example("I am not sure if I love you.", "negative"), 
  Example("Everything is fine.", "negative"), 
  Example("Okayy", "positive"), 
]
# texts that...

inputs=[
  "I will have to cancel for tonight",
]

response = co.classify(
  model='large',
  inputs=inputs,
  examples=examples,
)

print(response.classifications)

