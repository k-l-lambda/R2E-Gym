from pydantic import BaseModel, Field


class LLMArgs(BaseModel):
    """LLM configuration arguments."""
    model: str = Field("default", description="Model name")
    temperature: float = Field(0.0, description="Sampling temperature")
    max_tokens: int = Field(4096, description="Maximum tokens")
    base_url: str = Field("http://localhost:8000/v1", description="API base URL")


class LLMCompletions:
    """LLM completion helper."""

    @staticmethod
    def get_llm_completions(args, payloads):
        import litellm

        results = []
        for payload in payloads:
            response = litellm.completion(
                model=args.model,
                messages=payload,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_base=args.base_url,
            )
            results.append([response.choices[0].message.content])
        return results
