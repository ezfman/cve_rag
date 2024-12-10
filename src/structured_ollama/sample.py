from ollama import chat, Client
from pydantic import BaseModel


class Pet(BaseModel):
  name: str
  animal: str
  age: int
  color: str | None
  favorite_toy: str | None


class PetList(BaseModel):
  pets: list[Pet]


def test_client(client: Client):
    response = client.chat(
        model='qwen2.5', messages=[
            {
                'role': 'user',
                'content': 'Why is the sky blue?',
            },
        ]
    )

    print(response.message.content)


def main():
    client = Client(
    host='http://192.168.68.68:11434',
    headers={'x-some-header': 'some-value'}
    )

    response = client.chat(
        messages=[
            {
            'role': 'user',
            'content': '''
                I have two pets.
                A cat named Luna who is 5 years old and loves playing with yarn. She has grey fur.
                I also have a 2 year old black cat named Loki who loves tennis balls.
            ''',
            }
        ],
        model='llama3.2:3b-instruct-q4_K_M',
        format=PetList.model_json_schema(),
    )

    pets = PetList.model_validate_json(response.message.content)
    print(pets)