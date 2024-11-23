from scipy.sparse import csr_matrix


def run_conversation_through_goodfire(conversation: dict, variant, client):
    variant.reset()

    response = ""

    print(f"Getting the response for the conversation: {conversation}")

    for token in client.chat.completions.create(
            conversation,
            model=variant,
            stream=True,
            max_completion_tokens=40,
    ):
        response += token.choices[0].delta.content

    print(f"The response was: {response}\nGetting the activations for the same prompt")
    context = client.features.inspect(
        conversation,
        model=variant,
    )
    activations = context.matrix(return_lookup=False)
    dense_activations = csr_matrix(activations)

    return {
        "prompt": conversation,
        "response": response,
        "features": dense_activations,
    }