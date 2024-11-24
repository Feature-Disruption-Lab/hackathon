from scipy.sparse import csr_matrix
import goodfire

def run_conversation_through_goodfire(conversation: dict, variant: goodfire.Variant, client: goodfire.Client, activations=True, max_tokens=40) -> dict:
    variant.reset()

    response = ""

    # print(f"Getting the response for the conversation: {conversation}")

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
        "prompt": conversation
        "response": response
        }

def clone_variant(variant: goodfire.Variant):
    new_variant = goodfire.Variant(variant.base_model)
    for edit in variant.edits:
        new_variant.set(edit[0], edit[1]['value'], mode=edit[1]['mode'])

    return new_variant
