import sys

from gpt4all import GPT4All


PARAMS = dict(
    temp=0.9,
    top_k=1_000,
    top_p=0.7,
    n_batch=1024,
    max_tokens=1_000,
)


def do_setup():
    print('Loading...')
    model = GPT4All(
        'mistral-7b-instruct-v0.2.Q5_K_M.gguf',
        model_path='./',
        allow_download=False)

    philosophy = input('Provide a philosophy for your therapist: ')

    print()

    system_prompt = f"""\
# INSTRUCTIONS:

You are a therapist. You help patients explore their feelings and desires. You acknowledge what patients say and make them feel heard. You ask questions to explore details. You offer advice on what they should do. Begin by introducing yourself. You will follow this philosophy for therapy: "{philosophy}"
"""

    return system_prompt, model


STOP_GENERATING = False


def end_turn(token_id, token_string):
    if STOP_GENERATING:
        return False

    if '#' in token_string:
        return False

    return True


def print_response(response_it):
    for response in response_it:
        print(response, end='')
        sys.stdout.flush()

    print()


def do_loop(system_prompt, model):
    with model.chat_session(
            system_prompt='Use markdown formatting for all input and output. Do not output JSON or HTML tags.',
            prompt_template='{0}'):

        response_it = model.generate(
            prompt=f'{system_prompt}\n\n# THERAPIST:\n',
            streaming=True,
            callback=end_turn,
            **PARAMS)
        print_response(response_it)

        while True:
            print()
            prompt = input('> ')
            if prompt == 'quit':
                break
            if not prompt:
                continue

            print()
            response_it = model.generate(
                prompt=f"# PATIENT:\n{prompt}\n\n# THERAPIST:\n",
                streaming=True,
                callback=end_turn,
                **PARAMS)
            print_response(response_it)


def main():
    global STOP_GENERATING
    try:
        system_prompt, model = do_setup()
        do_loop(system_prompt, model)
    except KeyboardInterrupt:
        STOP_GENERATING = True
        print('Exiting...')

    sys.exit(0)


if __name__ == '__main__':
    main()
