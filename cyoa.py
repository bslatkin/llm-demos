import sys

from gpt4all import GPT4All


PARAMS = dict(
    temp=0.9,
    top_k=100,
    top_p=0.6,
    n_batch=1024,
    max_tokens=1_000,
)


def do_setup():
    print('Loading...')
    model = GPT4All(
        'mistral-7b-instruct-v0.2.Q5_K_M.gguf',
        model_path='./',
        allow_download=False)

    setting = input('Provide a setting for a "choose your own adventure" story: ')

    print()

    system_prompt = f"""\
# INSTRUCTIONS:

You are a text-based game with the following plot: "{setting}"

During the game you will describe each scene to the player and what they are seeing. You will tell them details about each of the people in the scene, what they look like, how they act, and what they might be thinking privately. You will not make choices for the player or determine anything they do. Instead, you will ask the player to decide what to do next each time. You will not present an explicit list of options but allow the player to make an unstructured choice.
# """

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
            prompt_template='{0}'):

        response_it = model.generate(
            prompt=(
                f"{system_prompt}\n\n"
                "# PLAYER'S COMMAND:\nDescribe the scene\n\n"
                "# GAME:\n"),
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
                prompt=f"# PLAYER'S COMMAND:\n{prompt}\n\n# GAME:\n",
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
