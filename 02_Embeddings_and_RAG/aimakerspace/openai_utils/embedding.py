from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, RateLimitError
import openai
from typing import List
import os
import asyncio
import time
import random


class EmbeddingModel:
    def __init__(self, embeddings_model_name: str = "text-embedding-3-small"):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.async_client = AsyncOpenAI()
        self.client = OpenAI()

        if self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key."
            )
        openai.api_key = self.openai_api_key
        self.embeddings_model_name = embeddings_model_name

    async def _retry_with_backoff(self, func, max_retries=5, base_delay=1):
        """Retry function with exponential backoff for rate limit errors"""
        for attempt in range(max_retries):
            try:
                return await func()
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Extract wait time from error message if available
                error_msg = str(e)
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                # Try to parse wait time from error message
                if "Please try again in" in error_msg:
                    try:
                        wait_part = error_msg.split("Please try again in ")[1].split("s.")[0]
                        suggested_wait = float(wait_part)
                        wait_time = max(wait_time, suggested_wait + random.uniform(1, 3))
                    except:
                        pass
                
                print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(base_delay * (2 ** attempt))

    async def async_get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        # Reduce batch size to avoid hitting rate limits as frequently
        batch_size = 512  # Reduced from 1024
        batches = [list_of_text[i:i + batch_size] for i in range(0, len(list_of_text), batch_size)]
        
        async def process_batch_with_retry(batch, batch_index):
            async def batch_func():
                embedding_response = await self.async_client.embeddings.create(
                    input=batch, model=self.embeddings_model_name
                )
                return [embeddings.embedding for embeddings in embedding_response.data]
            
            result = await self._retry_with_backoff(batch_func)
            print(f"Processed batch {batch_index + 1}/{len(batches)} ({len(batch)} texts)")
            return result
        
        # Process batches with some delay to avoid overwhelming the API
        results = []
        for i, batch in enumerate(batches):
            if i > 0:  # Add small delay between batches (except first)
                await asyncio.sleep(0.5)
            
            batch_result = await process_batch_with_retry(batch, i)
            results.append(batch_result)
        
        # Flatten the results
        return [embedding for batch_result in results for embedding in batch_result]

    async def async_get_embedding(self, text: str) -> List[float]:
        async def embedding_func():
            embedding = await self.async_client.embeddings.create(
                input=text, model=self.embeddings_model_name
            )
            return embedding.data[0].embedding
        
        return await self._retry_with_backoff(embedding_func)

    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        embedding_response = self.client.embeddings.create(
            input=list_of_text, model=self.embeddings_model_name
        )

        return [embeddings.embedding for embeddings in embedding_response.data]

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.client.embeddings.create(
            input=text, model=self.embeddings_model_name
        )

        return embedding.data[0].embedding


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    print(asyncio.run(embedding_model.async_get_embedding("Hello, world!")))
    print(
        asyncio.run(
            embedding_model.async_get_embeddings(["Hello, world!", "Goodbye, world!"])
        )
    )
