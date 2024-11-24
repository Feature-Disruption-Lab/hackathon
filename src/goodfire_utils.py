from scipy.sparse import csr_matrix


def run_conversation_through_goodfire(conversation: dict, variant, client, activations=True, max_tokens=40):
    variant.reset()

    response = ""

    print(f"Getting the response for the conversation: {conversation}")

    for token in client.chat.completions.create(
            conversation,
            model=variant,
            stream=True,
            max_completion_tokens=max_tokens,
    ):
        response += token.choices[0].delta.content

    context = client.features.inspect(
        conversation,
        model=variant,
    )
    if activations:
        print("Getting the activations for the same prompt")
        activations = context.matrix(return_lookup=False)
        dense_activations = csr_matrix(activations)

        return {
            "prompt": conversation,
            "response": response,
            "features": dense_activations,
        }
    else:
        print(f"The response was: {response}")
        return {
        "prompt": conversation,
        "response": response
        }