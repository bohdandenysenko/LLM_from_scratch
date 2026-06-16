from src.arc.model import GPTModel
from src.arc.config import GPTConfig
import tiktoken
import torch

from src.utils.generation import generate_text, load_hf_gpt2_weights


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    model_config = GPTConfig(qkv_bias=True)
    gpt2_model = GPTModel(config=model_config).to(device)

        
    

    # Load the weights
    gpt2_model = load_hf_gpt2_weights(custom_model=gpt2_model, model_name="gpt2")
    # print(gpt2_model.__dict__)

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = "Openchip hands-on workshops are awesome because"

    output = generate_text(
        model=gpt2_model,
        tokenizer=tokenizer,
        text=prompt,
        max_new_tokens=54,
        temperature=0.4,
        top_k=40,
        context_length=model_config.context_length,
        device=device,
    )

    print(f"\nGenerated text:\n{output}")