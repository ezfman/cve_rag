import gzip
import json
import os

from httpx import Client as RESTClient
from numpy import dot
from numpy.linalg import norm
from ollama import Client
from pydantic import BaseModel#, conlist
from tqdm import tqdm


class OllamaClient:
    def __init__(self, host: str = "http://192.168.68.68:11434"):
        self.client = Client(host=host, headers={"x-some-header": "some-value"})

    def embed(self, prompt: str, model: str = "nomic-embed-text"):
        resp = self.client.embeddings(model=model, prompt=prompt)
        return resp.embedding

    def prompt(
        self,
        prompt: str,
        system_promt: str = None,
        model: str = "qwen2.5",
        role: str = "user",
    ):
        resp = self.client.chat(
            messages=[
                {
                    "role": role,
                    "content": prompt,
                }
            ],
            model = model,
            format = CVE.model_json_schema(),
        )

        response = resp.message.content

        return CVERemediations.model_validate_json(response)


class CVEScraper:
    def __init__(self, client: OllamaClient):
        if os.path.exists("nvdcve-1.1-modified.json"):
            self.data = json.load(open("nvdcve-1.1-modified.json"))
        else:
            client = RESTClient()
            resp = client.get(
                "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-modified.json.gz"
            )
            # TODO: Handle status codes
            if resp.status_code != 200:
                raise ValueError("ruh roh")

            raw_data = gzip.decompress(resp.content)
            self.data = json.loads(raw_data)

        self.cve_count = len(self.data["CVE_Items"])
        self._parser(client)

    def _parser(self, client: OllamaClient):
        self.cve_list = []
        self.embeddings = []

        for cve in tqdm(self.data["CVE_Items"], total=self.cve_count):
            description = cve["cve"]["description"]["description_data"][0]["value"]
            self.cve_list.append(
                CVE(
                    id=cve["cve"]["CVE_data_meta"]["ID"],
                    description=description,
                    # publishedDate = cve['publishedDate'],
                    # lastModifiedDate = cve['lastModifiedDate'],
                )
            )
            self.embeddings.append(client.embed(description))


class CVE(BaseModel):
    id: str
    description: str
    # embedding: conlist(float, min_length=768, max_length=768)  # type: ignore
    # publishedDate: str
    # lastModifiedDate: str


class CVERemediation(BaseModel):
    cve_id: str
    remediation_strategy: str
    helpful_datasets: list[str]
    aggregations: str


class CVERemediations(BaseModel):
    cves: list[CVERemediation]


def test_client(client: Client):
    response = client.chat(
        model="qwen2.5",
        messages=[
            {
                "role": "user",
                "content": "Why is the sky blue?",
            },
        ],
    )

    print(response.message.content)


def embed_text(text: str, client: Client, model: str = "nomic-embed-text"):
    resp = client.embeddings(model=model, prompt=text)
    return resp.embedding


def semantic_search(prompt: str, scraper: CVEScraper, client: OllamaClient):
    prompt_embedding = client.embed(prompt)
    prompt_norm = norm(prompt_embedding)

    similarity_scores = [
        dot(prompt_embedding, embedding) / (prompt_norm * norm(embedding))
        for embedding in scraper.embeddings
    ]
    return sorted(zip(similarity_scores, range(scraper.cve_count), scraper.cve_list), reverse=True)


def main():
    ollama_client = OllamaClient()
    scraper = CVEScraper(ollama_client)

    # cve_list = scraper.cve_list
    # embeddings = scraper.embeddings
    # assert len(cve_list == len(embeddings))

    user_input = input("What happened? ")

    rag_hits = semantic_search(user_input, scraper, ollama_client)

    rag_prompt = "\n".join([f"{n+1}. {hit}" for n, hit in enumerate(rag_hits[:5])])
    print(rag_prompt)

    import IPython; IPython.embed()

    # client = None
    # response = client.chat(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": user_input,
    #         }
    #     ],
    #     model="llama3.2:3b-instruct-q4_K_M",
    #     format=CVEList.model_json_schema(),
    # )

    # pets = CVEList.model_validate_json(response.message.content)
    # print(pets)


main()